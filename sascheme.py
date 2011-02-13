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
           len(tokens) > 2 and \
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
    def __init__(return_type, value):
        Function.__init__(self, return_type, [])
        this.value = value

    def apply(self, xs=None):
        return this.value


####


class SType(object): 
    type_name = "type"
    def __init__(self, value, token=None):
        self.value = value
        self.token = token
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return '<SType type_name="%s" value="%s" token="%s" />' % (self.type_name, self.value, self.token) 


class ParenType(SType):
    type_name = "paren"
    def is_open_paren(self):
        return "(" == self.value
    def is_close_paren(self):
        return ")" == self.value
    @staticmethod
    def can_box_value(value):
        return PareType.can_box_token(value)
    @staticmethod
    def can_box_token(token):
        return "(" == token or ")" == token
    @staticmethod
    def from_token(token):
        return ParenType(token, token)


class BoolType(SType):
    type_name = "bool"
    def unbox_bool(self):
        return bool(self.value)
    @staticmethod
    def can_box_value(value):
        return True
    @staticmethod
    def can_box_token(token):
        return "True" == token or "False" == token
    @staticmethod
    def from_token(token):
        return BoolType(bool(token), token)


class NumType(BoolType):
    type_name = "num"
    def unbox_num(self):
        return self.value
    @staticmethod
    def cast_value(value):
        try:
            return int(value)
        except ValueError:
            return float(value)
    @staticmethod
    def can_box_value(value):
        try:
            NumType.cast_value(value)
            return True
        except ValueError:
            return False
    @staticmethod
    def can_box_token(token):
        return NumType.can_box_value(token)
    @staticmethod
    def from_token(token):
        return NumType(NumType.cast_value(token), token)


class SymType(SType):
    type_name = "sym"
    @staticmethod
    def can_box_token(token):
        return not BoolType.can_box_token(token) and \
               not NumType.can_box_token(token)
    @staticmethod
    def can_box_value(value):
        return SymType.can_box_token(value)
    @staticmethod
    def from_token(token):
        return SymType(token, token)


####


class BasicTypeResolver(object):
    types = [
        ("paren", ParenType),
        ("bool", BoolType),
        ("num", NumType),
        ("sym", SymType)
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


def op_add(a, b): return NumType(a.unbox_num()+b.unbox_num())
def op_sub(a, b): return NumType(a.unbox_num()-b.unbox_num())
def op_mul(a, b): return NumType(a.unbox_num()*b.unbox_num())
def op_div(a, b): return NumType(a.unbox_num()/b.unbox_num())
def op_mod(a, b): return NumType(a.unbox_num()%b.unbox_num())
def op_neg(a):    return NumType(-a.unbox_num())
def op_not(a):    return BoolType(not a.unbox_bool())
def op_and(a, b): return BoolType(a.unbox_bool() and b.unbox_bool())
def op_or(a, b):  return BoolType(a.unbox_bool() or b.unbox_bool())
def op_gt(a, b):  return BoolType(a.unbox_bool() > b.unbox_bool())
def op_lt(a, b):  return BoolType(a.unbox_bool() < b.unbox_bool())
def op_eq(a, b):  return BoolType(a.unbox_bool() == b.unbox_bool())
def op_gte(a, b): return BoolType(a.unbox_bool() >= b.unbox_bool())
def op_lte(a, b): return BoolType(a.unbox_bool() <= b.unbox_bool())
def op_neq(a, b): return BoolType(a.unbox_bool() != b.unbox_bool())


class BasicFunctionResolver(object):
    builtin_functions = {
        "+"     : NaryFunction(NumType, [NumType, NumType], op_add),
        "-"     : NaryFunction(NumType, [NumType, NumType], op_sub),
        "*"     : NaryFunction(NumType, [NumType, NumType], op_mul),
        "/"     : NaryFunction(NumType, [NumType, NumType], op_div),
        "%"     : NaryFunction(NumType, [NumType, NumType], op_mod),
        "neg"   : NaryFunction(NumType, [NumType], op_neg),
        "not"   : NaryFunction(BoolType, [BoolType, BoolType], op_not),
        "and"   : NaryFunction(BoolType, [BoolType, BoolType], op_and),
        "or"    : NaryFunction(BoolType, [BoolType, BoolType], op_or),
        ">"     : NaryFunction(BoolType, [BoolType, BoolType], op_gt),
        "<"     : NaryFunction(BoolType, [BoolType, BoolType], op_lt),
        "=="    : NaryFunction(BoolType, [BoolType, BoolType], op_eq),
        ">="    : NaryFunction(BoolType, [BoolType, BoolType], op_gte),
        "<="    : NaryFunction(BoolType, [BoolType, BoolType], op_lte),
        "!="    : NaryFunction(BoolType, [BoolType, BoolType], op_neq)
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


class Context(CustomFunctionResolver, BasicTypeResolver): pass


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
        if type(self.stype) == SymType:
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
        else:
            return self

    @staticmethod
    def from_stream(context, tokens):
        titer = iter(tokens)
        root = None
        while True:
            try:
                token = titer.next()
            except StopIteration:
                raise Exception("Unexpedted end of tokens")

            stype = context.box_token(token)
            if root is None:
                if type(stype) == ParenType:
                    if stype.is_open_paren():
                        continue
                    else:
                        raise Exception("Unexpected closing paren")
                else:
                    node = ASTNode(stype)
                    root = node
                    continue
            else:
                if type(stype) == ParenType:
                    if stype.is_open_paren():
                        child = ASTNode.from_stream(context, titer)
                        root.children.append(child)
                    else:
                        return root
                else:
                    node = ASTNode(stype)
                    root.children.append(node)


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
