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
           is_balanced(tokens) and \
           len(tokens) > 2

#######
def func_mul(az): 
    it = iter(az)
    acc = float(it.next().token)
    for a in it:
        acc *= float(a.token)
    return acc

#def func_div(a, b): return a/b
def func_add(az):
    it = iter(az)
    acc = float(it.next().token)
    for a in it:
        acc += float(a.token)
    return acc

#def func_sub(a, b): return a-b
#def func_neg(a): return -a
#def func_mod(a, b): return a%b
#def func_not(a): return not a
#def func_and(a, b): return a and b
#def func_or(a, b): return a or b
#def func_gt(a, b): return a > b
#def func_lt(a, b): return a < b
#def func_eq(a, b): return a == b
#def func_gte(a, b): return a >= b
#def func_lte(a, b): return a <= b
#def func_neq(a, b): return a != b

########

class Context(object):
    types = set(["func", "num", "bool", "(", ")", "?"])
    func_map = {
        "*" : func_mul,
#        "/" : func_div,
        "+" : func_add,
#        "-" : func_sub,
#        "neg" : func_neg,
#        "%" : func_mod,
#        "not" : func_not,
#        "and" : func_and,
#        "or" : func_or,
#        ">" : func_gt,
#        "<" : func_lt,
#        "==" : func_eq,
#        ">=" : func_gte,
#        "<=" : func_lte,
#        "!=" : func_neq
    }

    def get_type(self, token):
        if token == "(" or token == ")":
            return token
        elif token == "True" or token == "False":
            return "bool"
        elif token in self.func_map.keys():
            return "func"
        else:
            try:
                num = float(token)
                return "num"
            except ValueError:
                return "?"

    def get_mapped_func(self, func):
        return self.func_map[func]

class ASTNode(object):
    def __init__(self, context, token):
        self.token = token
        self.token_type = context.get_type(token)
        self.parent = None
        self.children = []

    def is_well_typed(self):
        return self.token_type != "?" and \
               all(child.is_well_typed() for child in self.children)

    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return self.nodestr()

    def nodestr(self, depth=0):
        start = '<ASTNode token="%s" type="%s">' % (self.token, self.token_type)
        end = '</ASTNode>'
        if len(self.children) == 0:
            return "\t"*depth+start + end
        else:
            s = "\t"*depth+start+"\n"
            child_strs = (child.nodestr(depth+1) for child in self.children)
            for line in child_strs:
                s += ("\t"*(depth))+line+"\n"
            s += "\t"*depth+end
            return s

    def evaluate(self, context):
        if self.token_type == "func":
            func = context.get_mapped_func(self.token)
            ec_iter = (child.evaluate(context) for child in self.children)
            new_token = func(ec_iter)
            return ASTNode(context, new_token)
        else:
            return self

    @staticmethod
    def from_stream(context, tokens):
        titer = iter(tokens)
        depth = 0
        root = None
        while True:
            try:
                token = titer.next()
            except StopIteration:
                raise Exception("End of tokens - must be unbalanced")

            if root is None and token == "(":
                continue

            node = ASTNode(context, token)
            if root is None and node.token_type != "func":
                raise Exception("invalid syntax - no func")
            elif root is None and node.token_type == "func": 
                root = node
                continue

            else: # root is not None
                if node.token_type == "(":
                    child = ASTNode.from_stream(context, titer)
                    root.children.append(child)
                elif node.token_type == ")":
                    return root
                else:
                    root.children.append(node)

        raise Exception("no ending paren")


def execute(program):
    tokens = list(tokenize(program))
    print "tokens ", tokens
    print "is valid? ", is_valid(tokens)
    context = Context()
    tree = ASTNode.from_stream(context, tokens)
    print tree
    return tree.evaluate(context)

if __name__ == "__main__":
    import sys
    use_string = "USE: ./lisp.py (-f file | -c string)" 
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
    ret_val = execute(program)
    print "=== "+str(ret_val)
