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
        args = [xs.next().stype for i in xrange(self.arity())]
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

class NoneSType(BoxedSType):
    name = "none"
    def __init__(self):
        BoxedSType.__init__(self, None, None)

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

class MetaSymSType(MetaBoxedSType):
    def can_box_token(self, token):
        return not BoolSType.can_box_token(token) and \
               not NumSType.can_box_token(token)
    def can_box_value(self, value):
        return self.can_box_token(value)
    def from_token(self, token):
        return SymSType(token, token)

class SymSType(BoxedSType):
    __metaclass__ = MetaSymSType
    name = "sym"

class StreamSType(SType):
    name = "stream"
    def __init__(self, stream):
        self.stream = list(stream)
        SType.__init__(self, " ".join(self.stream))
    def to_ast_node(self, context):
        return ASTNode.from_stream(context, self.stream, force_eval=True)

####


class BasicTypeBoxer(object):
    types = [
        ("paren",   ParenSType),
        ("bool",    BoolSType),
        ("num",     NumSType),
        ("sym",     SymSType),
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
        s += tabs(depth+1)+type_str+"\n"
        child_strs = [child.nodestr(depth+2) for child in self.children]
        if len(child_strs) > 0:
            s += tabs(depth+1)+"<ASTNode.children>\n"
        for line in child_strs:
            s += line+"\n"
        if len(child_strs) > 0:
            s += tabs(depth+1)+"</ASTNode.children>\n"
        s += tabs(depth)+end
        return s

    def evaluate(self, context, depth=0):
        if type(self.stype) == SymSType:
            function = context.get_function(self.stype.value)
            ec_iter = (child.evaluate(context, depth+1) for child in self.children)
            new_stype = function.apply(ec_iter)
            new_node = ASTNode(new_stype)
            print self.stype.token, "*"*(depth+1)
            print self
            print self.stype.token, "*"*(depth+1)
            print new_node
            print self.stype.token, "*"*(depth+1)
            return new_node
        elif type(self.stype) == StreamSType:
            print "STREAM", "*"*(depth+1)
            print self
            print "STREAM", "*"*(depth+1)
            new_node = ASTNode.from_stream(context, self.stype.stream)
            print new_node
            print "STREAM", "*"*(depth+1)
            return new_node.evaluate(context, depth)
        else:
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
        depth = 0
        root = None
        while True:
            try:
                token = titer.next()
            except StopIteration:
                raise Exception("Unexpected end of tokens")

            stype = context.box_token(token)
            
            if root is None and type(stype) == ParenSType and \
               not stype.is_open_paren() and depth == 0:
                raise Exception("No opening paren")

            if root is None:
                if type(stype) == ParenSType:
                    if stype.is_open_paren():
                        depth += 1
                        continue
                    else:
                        depth -= 1
                        if depth < 0:
                            raise Exception("Too many closing parens")
                        return ASTNode(NoneSType()) 
                else:
                    node = ASTNode(stype)
                    root = node
                    continue
            else:
                if type(stype) == ParenSType:
                    if stype.is_open_paren():
                        child = ASTNode.from_stream_lazy(push_back("(", titer))
                        root.children.append(child)
                    else:
                        return root
                else:
                    node = ASTNode(stype)
                    root.children.append(node)

def push_back(sym, stream):
    yield sym
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
