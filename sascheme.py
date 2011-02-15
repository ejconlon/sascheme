#!/usr/bin/env python

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


class Function(object):
    def __init__(self, return_type, argument_types):
        self.return_type = return_type
        self.argument_types = argument_types
    def arity(self):
        return len(self.argument_types)


class NaryFunction(Function):
    def __init__(self, return_type, argument_types, operation):
        Function.__init__(self, return_type, argument_types)
        self.operation = operation

    def apply(self, xs):
        xs = iter(xs)
        args = (xs.next().stype for i in xrange(self.arity()))
        return self.operation(*args)


class Constant(Function):
    def __init__(self, return_type, value):
        Function.__init__(self, return_type, [])
        self.value = value

    def apply(self, xs=None):
        return self.value


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
    builtin_functions = {
        "+"     : NaryFunction(NumSType, [NumSType, NumSType], op_add),
        "-"     : NaryFunction(NumSType, [NumSType, NumSType], op_sub),
        "*"     : NaryFunction(NumSType, [NumSType, NumSType], op_mul),
        "/"     : NaryFunction(NumSType, [NumSType, NumSType], op_div),
        "%"     : NaryFunction(NumSType, [NumSType, NumSType], op_mod),
        "neg"   : NaryFunction(NumSType, [NumSType], op_neg),
        "not"   : NaryFunction(BoolSType, [BoolSType], op_not),
        "and"   : NaryFunction(BoolSType, [BoolSType, BoolSType], op_and),
        "or"    : NaryFunction(BoolSType, [BoolSType, BoolSType], op_or),
        ">"     : NaryFunction(BoolSType, [NumSType, NumSType], op_gt),
        "<"     : NaryFunction(BoolSType, [NumSType, NumSType], op_lt),
        "=="    : NaryFunction(BoolSType, [SType, SType], op_eq),
        ">="    : NaryFunction(BoolSType, [NumSType, NumSType], op_gte),
        "<="    : NaryFunction(BoolSType, [NumSType, NumSType], op_lte),
        "!="    : NaryFunction(BoolSType, [SType, SType], op_neq)
    }
    
    def get_function(self, token):
        return self.builtin_functions[token]
    

class CustomFunctionResolver(BasicFunctionResolver):
    custom_functions = {}

    def get_function(self, token):
        try:
            return BasicFunctionResolver.get_function(self, token)
        except KeyError:
            return self.custom_functions[token]

    def set_function(self, token, function):
        self.custom_functions[token] = function 


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


    def evaluate(self, context, depth=0):
        if type(self.stype) == ListSType:
            if len(self.children) > 0:
                first_child = self.children[0]
                # evaluate first child from stream
                if type(first_child.stype) == StreamSType:
                    first_child = first_child.evaluate(context, depth+1)
                # if is identifier type, then do function applicaiton
                if type(first_child.stype) == IdentSType:
                    function = context.get_function(first_child.stype.value)
                    ec_iter = (child.evaluate(context, depth+1) for child in self.children[1:])
                    new_stype = function.apply(ec_iter)
                    new_node = ASTNode(new_stype)
                    ASTNode.print_change(first_child.stype.token, depth, self, new_node)
                    return new_node
                # else return a list node with list type info for children
                else:
                    ec_iter = (child.evaluate(context, depth+1) for child in self.children)
                    new_stype = ListSType((child.stype for child in ec_iter))
                    new_node = ASTNode(new_stype)
                    new_node.children = self.children
                    return new_node
        elif type(self.stype) == StreamSType:
            new_node = ASTNode.from_stream(context, self.stype.stream)
            ASTNode.print_change("STREAM", depth, self, new_node)
            return new_node.evaluate(context, depth)
        # not a stream or list to reify or function application?
        return self

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
                        child = ASTNode.from_stream_lazy(push_back("(", titer))
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

def execute(program):
    tokens = tokenize(program)
    context = Context()
    tree = ASTNode.from_stream(context, tokens)
    return tree.evaluate(context)


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
