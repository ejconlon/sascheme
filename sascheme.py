#!/usr/bin/env python

import collections
import itertools

def tokenize(stream):
    last = []
    for c in stream:
        if c == " " or c == "\t" or c =="\n":
            if len(last) > 0: 
                yield "".join(last)
                last = []
            continue
        elif c == "(" or c == ")":
            if len(last) > 0: 
                yield "".join(last)
                last = []
            yield c
        else:
            last.append(c)
    if len(last) > 0:
        yield "".join(last)

def is_balanced(tokens):
    depth = 0
    for t in tokens:
        if t == "(":
            depth += 1
        elif t == ")":
            depth -= 1
    return depth == 0

def is_valid(tokens):
    return tokens[0]  == "(" and \
           tokens[-1] == ")" and \
           len(tokens) >= 2 and \
           is_balanced(tokens)


####


class MetaSType(type): pass

class SType(object): 
    __metaclass__ = MetaSType
    name = "type"
    def __init__(self, token=None):
        self.token = token
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        s = '<'+self.__class__.__name__+' name="%s" ' % (self.name)
        if self.token is not None:
            s += 'token="%s" ' % (self.token) 
        return s + '/>'

class MetaBoxedSType(MetaSType):
    def can_box_value(self, value):
        raise Exception("OVERRIDE PLEASE")
    def can_box_token(self, token):
        raise Exception("OVERRIDE PLEASE")
    def from_token(self, token):
        raise Exception("OVERRIDE PLEASE")

class BoxedSType(SType):
    __metaclass__ = MetaBoxedSType
    name = "boxed"
    def __init__(self, value, token=None):
        SType.__init__(self, token)
        self.value = value
    def __str__(self):
        s = '<'+self.__class__.__name__+' name="%s" value="%s" ' % (self.name, self.value)
        if self.token is not None:
            s += 'token="%s" ' % (self.token) 
        return s + '/>'

#class NoneSType(BoxedSType):
#    name = "none"
#    def __init__(self):
#        BoxedSType.__init__(self, None, None)

class MetaParenSType(MetaBoxedSType):
    def can_box_value(self, value):
        return ParenSType.can_box_token(value)
    def can_box_token(self, token):
        return "(" == token or ")" == token
    def from_token(self, token):
        return ParenSType(token, token)

class ParenSType(BoxedSType):
    __metaclass__ = MetaParenSType
    name = "paren"
    def is_open_paren(self):
        return "(" == self.value
    def is_close_paren(self):
        return ")" == self.value


class MetaBoolSType(MetaBoxedSType):
    def can_box_value(self, value):
        return True
    def can_box_token(self, token):
        return "#t" == token or "#f" == token
    def from_token(self, token):
        if token == "#t":
            value = True
        elif token == "#f":
            value = False
        else:
            raise Exception("Invalid token: "+token)
        return BoolSType(value, token)

class BoolSType(BoxedSType):
    __metaclass__ = MetaBoolSType
    name = "bool"
    def __init__(self, value, token=None):
        if token is None:
            if value:
                token = "#t"
            else:
                token = "#f"
        BoxedSType.__init__(self, value, token)
    def unbox_bool(self):
        return bool(self.value)


class MetaNumSType(MetaBoolSType):
    def cast_value(self, value):
        try:
            return int(value)
        except ValueError:
            return float(value)
    def can_box_value(self, value):
        try:
            self.cast_value(value)
            return True
        except ValueError:
            return False
    def can_box_token(self, token):
        return self.can_box_value(token)
    def from_token(self, token):
        return NumSType(self.cast_value(token), token)

class NumSType(BoxedSType):
    __metaclass__ = MetaNumSType
    name = "num"
    def __init__(self, value, token=None):
        if token is None:
            token = str(value)
        BoxedSType.__init__(self, value, token)
    def unbox_bool(self):
        return bool(self.value)
    def unbox_num(self):
        return self.value

class MetaIdentSType(MetaBoxedSType):
    def can_box_token(self, token):
        return not BoolSType.can_box_token(token) and \
               not NumSType.can_box_token(token)
    def can_box_value(self, value):
        return self.can_box_token(value)
    def from_token(self, token):
        return IdentSType(token, token)

class IdentSType(BoxedSType):
    __metaclass__ = MetaIdentSType
    name = "ident"

class StreamSType(SType):
    name = "stream"
    def __init__(self, stream):
        self.stream = list(stream)
        SType.__init__(self, " ".join(self.stream))
    def to_ast_node(self, context):
        return ASTNode.from_stream(context, self.stream)

class ListSType(SType):
    name = "list"
    def __init__(self, child_stypes=None):
        SType.__init__(self)
        self.child_stypes = child_stypes
    def __str__(self):
        s = '<'+self.__class__.__name__+' name="%s"' % (self.name)
        if self.child_stypes is not None:
            s += '>\n'
            s += '\n'.join('\t'+str(cs) for cs in self.child_stypes)
            s += '\n'
            return s + '</'+self.__class__.__name__+'>'
        else:
            return s + '/>'

####

class Function(SType):
    def __init__(self, token):
        SType.__init__(self, token)
    def arity(self):
        return 0
    def type_apply(self, context, nodes):
        raise Exception("OVERRIDE PLEASE")

class ConstantFunction(Function):
    def __init__(self, token, stype_value):
        Function.__init__(self, token)
        self.stype_value = stype_value
    def type_apply(self, context, nodes):
        return self.stype_value

class LambdaFunction(Function):
    def __init__(self, token, stype_value, bindings):
        Function.__init__(self, token)
        self.stype_value = stype_value
        self.bindings = bindings
    def arity(self):
        return -1 # make arity checks fail
    def type_apply(self, context, nodes):
        head = ASTNode(self.stype_value)
        head.children = nodes
        return head.applied(context).stype

# lifts an stype* -> stype function to astnode* -> stype
class NaryFunction(Function):
    def __init__(self, token, return_type, argument_types, operation):
        Function.__init__(self, token)
        self.return_type = return_type
        self.argument_types = argument_types
        self.operation = operation
    def arity(self):
        return len(self.argument_types)
    def type_apply(self, context, nodes):
        nodes = list(nodes)
        print "nodes", nodes
        print self.arity()
        nargs = [nodes[i].applied(context) for i in xrange(self.arity())]
        print "nargs", nargs
        stypes = [narg.stype for narg in nargs]
        print "stypes", stypes
        return self.operation(*stypes)

####

class BasicTypeBoxer(object):
    types = [
        ("paren",   ParenSType),
        ("bool",    BoolSType),
        ("num",     NumSType),
        ("ident",     IdentSType),
    ]

    def box_token(self, token):
        for tname, tclass in self.types:
            if tclass.can_box_token(token):
                return tclass.from_token(token)
        raise Exception("Could not cast token: "+str(token))
    
    def box_value(self, value):
        for tname, tclass in self.types:
            if tclass.can_box_value(value):
                return tclass(value)
        raise Exception("Could not cast value: "+str(value))


#### NUMERIC / LOGICAL OPERATIONS ####


def op_add(a, b): return NumSType(a.unbox_num()+b.unbox_num())
def op_sub(a, b): return NumSType(a.unbox_num()-b.unbox_num())
def op_mul(a, b): return NumSType(a.unbox_num()*b.unbox_num())
def op_div(a, b): return NumSType(a.unbox_num()/b.unbox_num())
def op_mod(a, b): return NumSType(a.unbox_num()%b.unbox_num())
def op_neg(a):    return NumSType(-a.unbox_num())
def op_not(a):    return BoolSType(not a.unbox_bool())
def op_and(a, b): return BoolSType(a.unbox_bool() and b.unbox_bool())
def op_or(a, b):  return BoolSType(a.unbox_bool() or b.unbox_bool())
def op_gt(a, b):  return BoolSType(a.unbox_num() > b.unbox_num())
def op_lt(a, b):  return BoolSType(a.unbox_num() < b.unbox_num())
def op_eq(a, b):  return BoolSType(a.value == b.value)
def op_gte(a, b): return op_or(op_gt(a,b), op_eq(a,b))
def op_lte(a, b): return op_or(op_lt(a,b), op_eq(a,b))
def op_neq(a, b): return op_not(op_eq(a,b))


class BasicFunctionResolver(object):
    builtin_functions = dict((func.token, func) for func in [
        NaryFunction("+", NumSType, [NumSType, NumSType], op_add),
        NaryFunction("-", NumSType, [NumSType, NumSType], op_sub),
        NaryFunction("*", NumSType, [NumSType, NumSType], op_mul),
        NaryFunction("/", NumSType, [NumSType, NumSType], op_div),
        NaryFunction("%", NumSType, [NumSType, NumSType], op_mod),
        NaryFunction("neg", NumSType, [NumSType], op_neg),
        NaryFunction("not", BoolSType, [BoolSType], op_not),
        NaryFunction("and", BoolSType, [BoolSType, BoolSType], op_and),
        NaryFunction("or", BoolSType, [BoolSType, BoolSType], op_or),
        NaryFunction(">", BoolSType, [NumSType, NumSType], op_gt),
        NaryFunction("<", BoolSType, [NumSType, NumSType], op_lt),
        NaryFunction("==", BoolSType, [SType, SType], op_eq),
        NaryFunction(">=", BoolSType, [NumSType, NumSType], op_gte),
        NaryFunction("<=", BoolSType, [NumSType, NumSType], op_lte),
        NaryFunction("!=", BoolSType, [SType, SType], op_neq)
    ])

    def get_function(self, token):
        return self.builtin_functions[token]
    

class CustomFunctionResolver(BasicFunctionResolver, Function):
    def __init__(self):
        BasicFunctionResolver.__init__(self)
        self.func_name = "define"
        Function.__init__(self, self.func_name)
        self.custom_functions = {}
        self.custom_variables = {}

    def arity(self):
        return 2

    def type_apply(self, context, nodes):
        #print "type_apply"
        nodes = iter(nodes)
        from_node = nodes.next()
        to_node = nodes.next()
        if type(from_node.stype) == IdentSType:
            token = from_node.stype.token
            self.set_variable(token,
                ConstantFunction(token, to_node.strict(context).stype))
        elif type(from_node.stype) == StreamSType:
            token = from_node.stype.stream[1]
            bindings = from_node.stype.stream[1:-1]
            if token == ")":
                raise Exception("Invalid define pattern in to: "+str(from_node))
            else:
                self.set_function(token, LambdaFunction(token, to_node.stype, bindings))
        else:
            raise Exception("Invalid define pattern in from: "+str(from_node))
        return ListSType()

    def get_function(self, token):
        try:
            return BasicFunctionResolver.get_function(self, token)
        except KeyError:
            if token == self.func_name:
                return self
            return self.custom_functions[token]

    def set_function(self, token, function):
        try:
            BasicFunctionResolver.get_function(self, token)
            raise Exception("Token already present in builtins: "+token)
        except KeyError:
            if token == self.func_name:
                raise Exception("Token is def keyword: "+token)
            self.custom_functions[token] = function 

    def get_variable(self, token):
        return self.custom_variables[token]

    def set_variable(self, token, function):
        self.custom_variables[token] = function

    def get_by_identifier(self, token):
        try:
            return self.get_function(token)
        except KeyError:
            return self.get_variable(token)


class Context(CustomFunctionResolver, BasicTypeBoxer): pass


####


class ASTNode(object):
    def __init__(self, stype):
        self.stype = stype
        self.parent = None
        self.children = []

    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return self.nodestr()

    def is_strict(self):
        return type(self.stype) != StreamSType

    def strict(self, context):
        if not self.is_strict():
            return ASTNode.from_stream(context, self.stype.stream)
        else:
            return self

    def applied(self, context):
        strict_self = self.strict(context)
        head_stype = strict_self.stype
        tail_nodes = strict_self.children
        if type(head_stype) == ListSType:
            if len(tail_nodes) > 0 and \
               type(tail_nodes[0].stype) == IdentSType:
                head_stype = tail_nodes[0].stype
                tail_nodes = tail_nodes[1:]
        if type(head_stype) == IdentSType:
            #print "applied"
            function = context.get_by_identifier(head_stype.token)
            #print "head_stype", head_stype
            #print "function", function
            new_stype = function.type_apply(context, tail_nodes)
            new_node = ASTNode(new_stype)
            #print "new_node", new_node
            ASTNode.print_change(head_stype.token, -1, self, new_node)
            return new_node.applied(context)
        return strict_self


    def nodestr(self, depth=0):
        start = '<ASTNode>' 
        type_str = str(self.stype)
        end = '</ASTNode>'
        tabs = lambda t: "\t"*t
        s =  tabs(depth)+start+"\n"
        for line in type_str.split('\n'):
            s += tabs(depth+1)+line+"\n"
        child_strs = [child.nodestr(depth+2) for child in self.children]
        if len(child_strs) > 0:
            s += tabs(depth+1)+"<ASTNode.children>\n"
        for line in child_strs:
            s += line+"\n"
        if len(child_strs) > 0:
            s += tabs(depth+1)+"</ASTNode.children>\n"
        s += tabs(depth)+end
        return s

    @staticmethod
    def print_change(token, depth, from_node, to_node):
        print token, "*"*(depth+1)
        print from_node 
        print token, "*"*(depth+1)
        print to_node
        print token, "*"*(depth+1)


    @staticmethod
    def from_stream_lazy(tokens):
        titer = iter(tokens)
        depth = 0
        buf = []

        while True:
            try:
                token = titer.next()
            except StopIteration:
                raise Exception("Unexpected end of tokens")

            if ParenSType.can_box_token(token):
                stype = ParenSType.from_token(token)
                
                if len(buf) == 0 and depth == 0 and not stype.is_open_paren():
                    raise Exception("No opening paren")

                if stype.is_open_paren():
                    depth += 1
                else:
                    depth -= 1
                    if depth == 0:
                        buf.append(token)
                        return ASTNode(StreamSType(buf))
                    elif depth < 0:
                        raise Exception("Too many closing parens")

            buf.append(token)

    @staticmethod
    def from_stream(context, tokens):
        titer = iter(tokens)
        root = None
        while True:
            try:
                token = titer.next()
            except StopIteration:
                raise Exception("Unexpected end of tokens")

            stype = context.box_token(token)

            if root is None and type(stype) == ParenSType and \
               not stype.is_open_paren():
                raise Exception("No opening paren")

            if root is None:
                # we raise exception above if the first token
                # is not an open paren
                root = ASTNode(ListSType())
                continue
            else:
                if type(stype) == ParenSType:
                    if stype.is_open_paren():
                        # open paren we make a lazy child
                        # and consume up to and including its close paren
                        child = ASTNode.from_stream_lazy(itertools.chain(["("], titer))
                        root.children.append(child)
                    else:
                        # close paren closes this element
                        return root
                else:
                    node = ASTNode(stype)
                    root.children.append(node)

def push_back(tok, stream):
    yield tok
    for s in stream: yield s


class LookAheadIterator(collections.Iterator):
    def __init__(self, wrapped):
        self._wrapped = iter(wrapped)
        self._need_to_advance = True
        self._has_next = False
        self._cache = None

    def has_next(self):
        if self._need_to_advance:
            self._advance()
        return self._has_next

    def _advance(self):
        try:
            self._cache = self._wrapped.next()
            self._has_next = True
        except StopIteration:
            self._has_next = False
        self._need_to_advance = False

    def next(self):
        if self._need_to_advance:
            self._advance()
        if self._has_next:
            self._need_to_advance = True
            return self._cache
        else:
            raise StopIteration()

    def __next__(self):
        self.next()

def execute(program):
    tokens = LookAheadIterator(tokenize(program))
    context = Context()
    ret_list = ASTNode(ListSType())
    while tokens.has_next():
        tree = ASTNode.from_stream(context, tokens)
        applied = tree.applied(context)
        ret_list.children.append(applied)
    return ret_list

if __name__ == "__main__":
    import sys
    use_string = "USE: sascheme.py (-f file | -c string)" 
    if len(sys.argv) < 3:
        print use_string
        sys.exit(-1)
    flag = sys.argv[1]
    arg = sys.argv[2]
    program = ""

    if flag == "-f":
        with open(arg, "r") as f:
            program = f.read()
    elif flag == "-c":
        program = arg
    else:
        print use_string
        sys.exit(-1)

    program = program.strip()
    ret_node = execute(program)
    print "==="
    print ret_node
