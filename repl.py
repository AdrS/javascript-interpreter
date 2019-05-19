#!/usr/bin/env python3

import enum

class TokenType(enum.Enum):
	VAR = 0
	FUNCTION = 1
	IF = 2
	WHILE = 3
	RETURN = 4
	IDENTIFIER = 5
	NUMBER = 6
	STRING = 7
	BOOL = 8
	NULL = 9
	OPEN_PAREN = 10
	CLOSE_PAREN = 11
	OPEN_BRACE = 12
	CLOSE_BRACE = 13
	OPEN_BRACKET = 14
	CLOSE_BRACKET = 15
	SEMICOLON = 16
	MUL_OP = 17
	ADD_OP = 18
	REL_OP = 19
	EQ_OP = 20
	NOT = 21
	ASSIGN_OP = 22
	COMMA = 23
	AND_OP = 24
	OR_OP = 25
	EOF = 26

class Lexer:
	reservedWords = {
		'true'   : (TokenType.BOOL,     True),
		'false'  : (TokenType.BOOL,     False),
		'null'   : (TokenType.NULL,     None),
		'var'    : (TokenType.VAR,      None),
		'function': (TokenType.FUNCTION, None),
		'if'     : (TokenType.IF,       None),
		'in'     : (TokenType.REL_OP,   'in'),
		'while'  : (TokenType.WHILE,    None),
		'return' : (TokenType.RETURN,   None)
	}

	operators = {
		'+': TokenType.ADD_OP,
		'-': TokenType.ADD_OP,
		'*': TokenType.MUL_OP,
		'/': TokenType.MUL_OP,
		'%': TokenType.MUL_OP,
		'<': TokenType.REL_OP,
		'>': TokenType.REL_OP,
		'!': TokenType.NOT,
		'=': TokenType.ASSIGN_OP,
		'+=': TokenType.ASSIGN_OP,
		'-=': TokenType.ASSIGN_OP,
		'*=': TokenType.ASSIGN_OP,
		'/=': TokenType.ASSIGN_OP,
		'%=': TokenType.ASSIGN_OP,
		'<=': TokenType.REL_OP,
		'>=': TokenType.REL_OP,
		'!=': TokenType.EQ_OP,
		'==': TokenType.EQ_OP,
		'&&': TokenType.AND_OP,
		'||': TokenType.OR_OP,
		',': TokenType.COMMA
	}

	punctuation = {
		'(': TokenType.OPEN_PAREN,
		')': TokenType.CLOSE_PAREN,
		'{': TokenType.OPEN_BRACE,
		'}': TokenType.CLOSE_BRACE,
		'[': TokenType.OPEN_BRACKET,
		']': TokenType.CLOSE_BRACKET,
		',': TokenType.COMMA,
		';': TokenType.SEMICOLON
	}

	escapeSequences = {
		'b': '\b',
		'f': '\f',
		'n': '\n',
		'r': '\r',
		't': '\t',
		'v': '\v',
		'0': '\0',
		'\'': '\'',
		'"': '"',
		'\\': '\\'
	}

	def __init__(self, s):
		self.s = s
		self.i = 0

	def getchar(self, noEof = False):
		pos = self.i
		self.i += 1
		if pos < len(self.s):
			return self.s[pos]
		else:
			if noEof:
				raise Exception('unexpected EOF')
			return ''

	def unget(self):
		self.i -= 1

	def lexan(self):
		while True:
			c = self.getchar()
			# EOF
			if c == '':
				return (TokenType.EOF, '')
			# Whitespace
			elif c.isspace():
				continue
			# Comments and Division
			elif c == '/':
				c = self.getchar()

				# Single line comment
				if c == '/':
					while c != '' and c != '\n':
						c = self.getchar()
				elif c == '*':
					# Multiline comments
					while True:
						c = self.getchar(True)
						while c != '*':
							c = self.getchar(True)
						c = self.getchar(True)
						if c == '/':
							break
				elif c == '=':
					return (Lexer.operators['/='], '/=')
				else:
					self.unget()
					return (Lexer.operators['/'], '/')
			# Numbers
			elif c.isdigit():
				# TODO: octal and hex literals
				# Integer Part
				number = 0
				while c.isdigit():
					number *= 10
					number += ord(c) - ord('0')
					c = self.getchar()

				# Check if floating point
				if c == '.':
					floatPart = 0
					decimalPlace = 0.1
					c = self.getchar(noEof=True)
					if not c.isdigit():
						raise Exception('invalid token %r at %d' % (c, i))

					while c.isdigit():
						floatPart += decimalPlace * (ord(c) - ord('0'))
						decimalPlace /= 10.0
						c = self.getchar()
					number += floatPart

				self.unget()
				return (TokenType.NUMBER, number)
			# Identifiers and reserved words
			elif c.isalpha() or c == '$' or c == '_':
				identifier = ''
				while c.isalnum() or c == '$' or c == '_':
					identifier += c
					c = self.getchar()
				self.unget()
				return Lexer.reservedWords.get(identifier, (TokenType.IDENTIFIER, identifier))
			# String literals
			elif c == '"' or c == '\'':
				quoteType = c
				text = ''
				c = self.getchar(noEof=True)
				while c != quoteType:
					# Escape sequence
					if c == '\\':
						c = self.getchar(noEof=True)
						c = Lexer.escapeSequences.get(c, c)
					text += c
					c = self.getchar(noEof=True)

				return (TokenType.STRING, text)
			# Operators
			elif c in ['<', '>', '=', '+', '-', '*', '%', '!']:
				if self.getchar() == '=':
					return (Lexer.operators[c + '='], c + '=')
				else:
					self.unget()
					return (Lexer.operators[c], c)
			elif c == '&':
				if self.getchar() != '&':
					raise Exception('invalid token for and')
				return (TokenType.AND_OP, '&&')
			elif c == '|':
				if self.getchar() != '|':
					raise Exception('invalid token for or')
				return (TokenType.OR_OP, '||')
			# Punctuation
			elif c in Lexer.punctuation:
				return (Lexer.punctuation[c], None)
			else:
				raise Exception('unknown token %r' % c)

	def tokenize(self):
		tokens = []
		while True:
			t = self.lexan()
			if t[0] == TokenType.EOF:
				break
			tokens.append(t)
		return tokens

class Expression:
	pass

class Number(Expression):
	def __init__(self, value):
		self.value = value
	def __repr__(self):
		return '%f' % self.value

class String(Expression):
	def __init__(self, value):
		self.value = value

	def __repr__(self):
		return '%r' % self.value

class Bool(Expression):
	def __init__(self, value):
		assert(value in [True, False])
		self.value = value
	def __repr__(self):
		if self.value:
			return 'true'
		return 'false'

class Null(Expression):
	def __repr__(self):
		return 'null'

class Identifier(Expression):
	def __init__(self, name):
		self.name = name
	def __repr__(self):
		return 'id(%s)' % self.name

class MemberAccess(Expression):
	def __init__(self, obj, member):
		assert(isinstance(member, Expression))
		self.obj, self.member = obj, member
	def __repr__(self):
		return '%r[%r]' % (self.obj, self.member)

class Call(Expression):
	def __init__(self, fun, args):
		for arg in args:
			assert(isinstance(arg, Expression))
		self.fun, self.args = fun, args
	def __repr__(self):
		args = ', '.join(a.__repr__() for a in self.args)
		return 'call(%r, %s)' % (self.fun, args)

class Block:
	def __init__(self, statements):
		self.statements = statements
	def __repr__(self):
		r = '\n'.join(s.__repr__() for s in self.statements)
		return '{ %s }' % r

class Function(Expression):
	def __init__(self, params, body, env=None):
		paramNames = set()
		for param in params:
			if param.name in paramNames:
				raise Exception('Repeated function parameter name %r' % param.name)
			paramNames.add(param.name)
		# TODO: how to handle environment (closures ...)
		self.params = params
		self.body = body

	def __repr__(self):
		paramsRep = ', '.join(p.__repr__() for p in self.params)
		return 'function(%s)%r' % (paramsRep, self.body)

class Not(Expression):
	def __init__(self, expr):
		assert(isinstance(expr, Expression))
		self.expr = expr
	def __repr__(self):
		return 'not(%r)' % self.expr

class UnaryMinus(Expression):
	def __init__(self, expr):
		assert(isinstance(expr, Expression))
		self.expr = expr
	def __repr__(self):
		return 'minus(%r)' % self.expr

class BinaryOp(Expression):
	def __init__(self, lhs, rhs, op):
		assert(isinstance(lhs, Expression))
		assert(isinstance(rhs, Expression))
		self.lhs, self.rhs, self.op = lhs, rhs, op
	def __repr__(self):
		return '(%s %r %r)' % (self.op, self.lhs, self.rhs)

class MulOp(BinaryOp):
	def __init__(self, lhs, rhs, op):
		assert(op in '*/%')
		BinaryOp.__init__(self, lhs, rhs, op)

class AddOp(BinaryOp):
	def __init__(self, lhs, rhs, op):
		assert(op in '+-')
		BinaryOp.__init__(self, lhs, rhs, op)

class RelOp(BinaryOp):
	def __init__(self, lhs, rhs, op):
		assert(op in ['<', '<=', '>', '>=', 'in'])
		BinaryOp.__init__(self, lhs, rhs, op)

class EqOp(BinaryOp):
	def __init__(self, lhs, rhs, op):
		assert(op in ['==', '!='])
		BinaryOp.__init__(self, lhs, rhs, op)

class AssignOp(BinaryOp):
	def __init__(self, lhs, rhs, op):
		assert(op in ['=', '+=', '*=', '/=', '%='])
		BinaryOp.__init__(self, lhs, rhs, op)

class Parser:
	def __init__(self, s):
		self.lexer = Lexer(s)
		self.tokenType, self.tokenValue = self.lexer.lexan()

	def match(self, c):
		if self.tokenType != c:
			raise Exception('expected token of type %r not %r (%r)' % (c, self.tokenValue, self.tokenValue))
		self.tokenType, self.tokenValue = self.lexer.lexan()

	def statement(self):
		'''
		Statement -> Declaration | Block | IfStatement | WhileStatement | ReturnStatement | Expression
		'''
		# TODO:
		if self.tokenType == TokenType.OPEN_BRACE:
			return self.block()
		return self.expression()

	def block(self):
		statements = []
		self.match(TokenType.OPEN_BRACE)
		while self.tokenType != TokenType.CLOSE_BRACE:
			statements.append(self.statement)
		self.match(TokenType.CLOSE_BRACE)
		return Block(statements)

	def identifier(self):
		i = Identifier(self.tokenValue)
		self.match(TokenType.IDENTIFIER)
		return i

	def function(self):
		'''
		Function -> 'function' '(' ParameterNames ')' Block
		ParameterNames -> epsilon | ParameterName {',' ParameterName }*
		ParameterName -> Identifier
		'''
		self.match(TokenType.FUNCTION)
		self.match(TokenType.OPEN_PAREN)

		parameters = []
		if self.tokenType == TokenType.IDENTIFIER:
			parameters.append(self.identifier())
			while self.tokenType == TokenType.COMMA:
				self.match(TokenType.COMMA)
				parameters.append(self.identifier())

		self.match(TokenType.CLOSE_PAREN)
		body = self.block()
		return Function(parameters, body)

	def atom(self):
		'''
		Atom -> Number | String | Identifier | 'true' | 'false' | 'null' | Function | '(' Expression ')'
		'''
		if self.tokenType == TokenType.NUMBER:
			e = Number(self.tokenValue)
			self.match(TokenType.NUMBER)
		elif self.tokenType == TokenType.STRING:
			e = String(self.tokenValue)
			self.match(TokenType.STRING)
		elif self.tokenType == TokenType.IDENTIFIER:
			return self.identifier()
		elif self.tokenType == TokenType.BOOL:
			e = Bool(self.tokenValue)
			self.match(TokenType.BOOL)
		elif self.tokenType == TokenType.NULL:
			e = Null()
			self.match(TokenType.NULL)
		elif self.tokenType == TokenType.FUNCTION:
			return self.function()
		else:
			self.match(TokenType.OPEN_PAREN)
			e = self.expression()
			self.match(TokenType.CLOSE_PAREN)
		return e

	def unit(self):
		'''
		Unit -> Atom | MemberAccess | FunctionCall
		MemberAccess -> Atom '[' Expression ']'
		FunctionCall -> Atom '(' Arguments ')'
		Arguments -> epsilon | Expression {',' Expression }*
		'''
		u = self.atom()
		# Member access and function calls can be chained
		while self.tokenType in [TokenType.OPEN_BRACKET, TokenType.OPEN_PAREN]:
			if self.tokenType == TokenType.OPEN_BRACKET:
				self.match(TokenType.OPEN_BRACKET)
				member = self.expression()
				self.match(TokenType.CLOSE_BRACKET)
				u = MemberAccess(u, member)
			else:
				self.match(TokenType.OPEN_PAREN)
				# Parse arguments
				args = []
				if self.tokenType != TokenType.CLOSE_PAREN:
					args.append(self.expression())
					while self.tokenType == TokenType.COMMA:
						self.match(TokenType.COMMA)
						args.append(self.expression())
				self.match(TokenType.CLOSE_PAREN)
				u = Call(u, args)
		return u

	def factor(self):
		'''
		Factor -> '!' Factor | '-' Factor | Unit
		'''
		# Right to left associative
		if self.tokenType == TokenType.NOT:
			self.match(TokenType.NOT)
			return Not(self.factor())

		if self.tokenType == TokenType.ADD_OP:
			if self.tokenValue == '-':
				self.match(TokenType.ADD_OP)
				return UnaryMinus(self.factor())
			elif self.tokenValue == '+':
				self.match(TokenType.ADD_OP)
				return self.factor()
			else:
				raise Exception("unknown add op")

		return self.unit()

	def expression(self):
		# TODO: comma expression (typically used in for loops)
		'''
		Assignment -> EqExpression AssignOp Expression
		AssignOp -> '=' | '+=' | '-=' | '*=' | '/=' | '%='

		EqExpression -> RelationalExpression { EqOp EqExpression }*
		EqOp -> '==' | '!='

		RelationalExpression -> ArithematicExpression { RelOp ArithematicExpression }*
		RelOp -> '<' | '<=' | '>' | '>=' | 'in'

		ArithematicExpression -> Term { AddOp Term }*
		AddOp -> '+' | '-'

		Term -> Factor { MulOp Factor }*
		MulOp -> '*' | '/' | '%'
		'''
		# Operators in order of increasing precedence
		# op token type, op class, isLeftAssociative
		precedenceLevels = [
			(TokenType.ASSIGN_OP, AssignOp, False),
			(TokenType.EQ_OP, EqOp, True),
			(TokenType.REL_OP, RelOp, True),
			(TokenType.ADD_OP, AddOp, True),
			(TokenType.MUL_OP, MulOp, True)
		]

		def level(i):
			if i == len(precedenceLevels):
				return self.factor()
			tokenType, opClass, isLeftAssociative = precedenceLevels[i]

			expr = level(i + 1)
			if isLeftAssociative:
				while self.tokenType == tokenType:
					op = self.tokenValue
					self.match(tokenType)
					rhs = level(i + 1)
					expr = opClass(expr, rhs, op)
			elif self.tokenType == tokenType:
				op = self.tokenValue
				self.match(tokenType)
				return opClass(expr, level(i), op)

			return expr

		return level(0)

	def parse(self):
		program = self.statement()
		self.match(TokenType.EOF)
		return program

def testParser():
	testCases = [
		('0',),
		('\'hello\'',),
		('a',),
		('true',),
		('false',),
		('null',),
		('function(){}',),
		('function(a){}',),
		('function(a, b, c){}',),
		('(1)',),
		('a["hi"]',),
		('a["users"]["adrian"]',),
		('a("hi")',),
		('a("hi", 4, 8)',),
		('a["callbacks"]["error"]()',),
		('a["callbacks"]["error"](404, "Not found")',),
		('function(){}()',),
		('a[function(){}]',),
		('a[function(){}()]',),
		('function(a, b, c){}(1, 2, 3)',),
		('!!!!true',),
		('- - -7',),
		('- + +7',),
		('-8',),
		('!-8',),
		('-!0',),
		('71 * -8 / +2 % 10',),
		('order["quantity"] * prices[order["item"]]', ),
		('2 * 3 + 4 < 5 - 6/7',),
		('order["item"] in prices',),
		('x1 > y1 == x2 > y2',),
		('a = b = c',),
		('b += a = 2 * 3 + 4 < 5 - 6/7',),
	]

	for testCase in testCases:
		source = testCase[0]
		print(source, Parser(source).parse())

	invalid = [
		('function()', 'function without body'),
		('function)', 'function without open paren'),
		('function(', 'function without close paren'),
		('function(a b){}', 'invalid paramenter list'),
		('function(7){}', 'invalid paramenter list'),
		('function(a, b,){}', 'invalid paramenter list'),
		('function(a, b, a){}', 'invalid parameter list'),
		('a("hi"', 'no closing paren in call'),
		('a("hi", 2, 3', 'no closing paren in call'),
		('a["hi"', 'no closing brace in member access'),
		('-', 'unary minus without argument'),
		('- + - -', 'unary minus without argument'),
		('+', 'unary plus without argument'),
		('!', 'not without argument'),
		('!!!', 'not without argument'),
		('71 * -8 / +2 %', 'no modulus'),
		('71 * -8 /', 'no divisor'),
		('71 *', 'not rhs'),
	]

	for source, description in invalid:
		try:
			Parser(source).parse()
			print('FAIL: expected error', source, description)
		except:
			pass

def testLexer():
	testCases = [
		# Numbers
		('0', [(TokenType.NUMBER, 0)]),
		('123', [(TokenType.NUMBER, 123)]),
		('0.234', [(TokenType.NUMBER, 0.234)]),
		# Reserved words
		('true', [(TokenType.BOOL, True)]),
		('false', [(TokenType.BOOL, False)]),
		('null', [(TokenType.NULL, None)]),
		('var', [(TokenType.VAR, None)]),
		('function', [(TokenType.FUNCTION, None)]),
		('if', [(TokenType.IF, None)]),
		('while', [(TokenType.WHILE, None)]),
		('return', [(TokenType.RETURN, None)]),
		# Identifiers
		('a', [(TokenType.IDENTIFIER, 'a')]),
		('a123', [(TokenType.IDENTIFIER, 'a123')]),
		('ABC_DEF', [(TokenType.IDENTIFIER, 'ABC_DEF')]),
		('$', [(TokenType.IDENTIFIER, '$')]),
		('$Money', [(TokenType.IDENTIFIER, '$Money')]),
		# String literals and escape sequences
		('"literal"', [(TokenType.STRING, 'literal')]),
		('"escape \\" sequence \\\\ tada!"', [(TokenType.STRING, 'escape " sequence \\ tada!')]),
		('"\\b\\f\\n\\r\\t\\v\\0\\\'\\"\\\\"', [(TokenType.STRING, '\b\f\n\r\t\v\0\'\"\\')]),
		('"\\a\\c\\d\\e\\g"', [(TokenType.STRING, 'acdeg')]),
		('\'literal\'', [(TokenType.STRING, 'literal')]),
		('\'escape \\" sequence \\\\ tada!\'', [(TokenType.STRING, 'escape " sequence \\ tada!')]),
		('\'\\b\\f\\n\\r\\t\\v\\0\\\'\\"\\\\\'', [(TokenType.STRING, '\b\f\n\r\t\v\0\'\"\\')]),
		('\'\\a\\c\\d\\e\\g\'', [(TokenType.STRING, 'acdeg')]),
		('\'\\\'\'', [(TokenType.STRING, '\'')]),
		# Comments
		('a//hello people', [(TokenType.IDENTIFIER, 'a')]),
		('//hello people\n123', [(TokenType.NUMBER, 123)]),
		# Multi-line Comments
		('a /*hello people\n1 2 3\nhi*/ b', [(TokenType.IDENTIFIER, 'a'), (TokenType.IDENTIFIER, 'b')]),
		('a /*hello * people\n1 2 3\nhi*/ b', [(TokenType.IDENTIFIER, 'a'), (TokenType.IDENTIFIER, 'b')]),
		('a /*hello /* people\n1 2 3\nhi*/ b', [(TokenType.IDENTIFIER, 'a'), (TokenType.IDENTIFIER, 'b')]),
		('a /*hello * //people\n1 2 3\nhi*/ b', [(TokenType.IDENTIFIER, 'a'), (TokenType.IDENTIFIER, 'b')]),
		('a//hello people', [(TokenType.IDENTIFIER, 'a')]),
		# Punctuation
		('(){};', [(TokenType.OPEN_PAREN, None), (TokenType.CLOSE_PAREN, None), (TokenType.OPEN_BRACE, None), (TokenType.CLOSE_BRACE, None), (TokenType.SEMICOLON, None)]),
		# Operators
	]

	for source, expected in testCases:
		got = Lexer(source).tokenize()
		if got != expected:
			print('FAIL', source, got, expected)

	invalid = [
		'.234',
		'123.',
		'093',
		'123a',
		'/*',
		'/*adsf*'
	]
	for source in invalid:
		try:
			got = Lexer(source).tokenize()
			print('FAIL: expected error', source)
		except:
			pass

'''
Grammar
-------
Program -> StatementList
Block -> '{' StatementList '}'
StatementList -> { Statement }*
Statement -> Declaration | Block | IfStatement | WhileStatement | ReturnStatement | Expression

Declaration -> 'var' Identifier [ '=' Expression ] ';'
IfStatement -> 'if' '(' Expression ')' Statement [ 'else' Statement ]
WhileStatement -> 'while '(' Expression ')' Statement
ReturnStatement -> 'return' [ Expression ] ';'

Expression -> EqExpression | Assignment

Assignment -> EqExpression '=' Expression

EqExpression -> RelationalExpression { EqOp EqExpression }*
EqOp -> '==' | '!='

RelationalExpression -> ArithematicExpression { RelOp ArithematicExpression }*
RelOp -> '<' | '<=' | '>' | '>=' | 'in'

ArithematicExpression -> Term { AddOp Term }*
AddOp -> '+' | '-'

Term -> Factor { MulOp Factor }*
MulOp -> '*' | '/' | '%'

Factor -> '!' Factor | '-' Factor | '+' Factor | Unit

Unit -> Atom | MemberAccess | FunctionCall

Atom -> Number | String | Identifier | 'true' | 'false' | 'null' | Function | '(' Expression ')'

FunctionCall -> Atom '(' Arguments ')'
Arguments -> epsilon | Expression {',' Expression }*

MemberAccess -> Atom '[' Expression ']'

Function -> 'function' '(' ParameterNames ')' Block
ParameterNames -> epsilon | ParameterName {',' ParameterName }*
ParameterName -> Identifier

see: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Operator_Precedence
'''
if __name__ == '__main__':
	testLexer()
	testParser()
