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
		',': TokenType.COMMA
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
			elif c in ['<', '>', '=', '+', '-', '*', '!']:
				if self.getchar() == '=':
					return (Lexer.operators[c + '='], c + '=')
				else:
					self.unget()
					return (Lexer.operators[c], c)
			elif c in Lexer.punctuation:
				return (Lexer.punctuation[c], None)
			elif c == '&':
				if self.getchar() != '&':
					raise Exception('invalid token for and')
				return (TokenType.AND_OP, '&&')
			elif c == '|':
				if self.getchar() != '|':
					raise Exception('invalid token for or')
				return (TokenType.OR_OP, '||')
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

class Parser:
	def __init__(self, s):
		self.lexer = Lexer(s)
		self.lookahead = None

	def match(self, c):
		pass

	def atom(self):
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

class Parser:
	def __init__(s):
		self.lexer = Lexer(s)

def testParser():
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
RelOp -> '<' | '<=' | '>' | '>='

ArithematicExpression -> Term { AddOp Term }*
AddOp -> '+' | '-'

Term -> Factor { MulOp Factor }*
MulOp -> '*' | '/' | '%'

Factor -> '!' Unit | '-' Unit | Unit

Unit -> Atom | MemberAccess | FunctionCall

Atom -> Number | String | Identifier | 'true' | 'false' | 'null' | Function | '(' Expression ')'

FunctionCall -> Expression '(' ParameterValues ')'
ParameterValues -> epsilon | Expression {',' Expression }*

MemberAccess -> Atom '[' Expression ']'

Function -> 'function' '(' ParameterNames ')' Block
ParameterNames -> epsilon | ParameterName {',' ParameterName }*
ParameterName -> Identifier

see: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Operator_Precedence
'''
if __name__ == '__main__':
	testLexer()
