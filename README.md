# JavaScript Interpreter

An interpreter for a (modified) subset of JavaScript written in Python.

# Examples of Supported Features
```javascript
/******** Basic Types ******************************/
// Numbers
console.log('numbers', 123, 3.14);

// Strings
console.log("double quotes", 'single quotes', 'escape\n\t\asequences');

// null (no undefined for simplicity)
var a;
console.log(a, null, a == null);

// Object literals
var origin = {x:0, y:0};
var p = {'x': 1, 'y':1};

// Bracket notation
var dx = origin['x'] - p['x'];

// Dot notation
var dy = origin.y - p.y;

/******** Functions ******************************/
// Closures and recursion
var fib = (function() {
	var memo = {};
	return function(n) {
		if(n < 2) {
			return n;
		} else if(n in memo) {
			return memo[n];
		} else {
			// No tail recursion optimization or named functions yet
			var f = fib(n - 1) + fib(n - 2);
			memo[n] = f;
			return f;
		}
	};
})();

// Higher order functions
var map = function(f, start, end, step) {
	var i = start;
	// While loops (for and do-while loops omitted for simplicity)
	while(i < end) {
		f(i);
		// +=, -=, *=, /=, %= operators supported
		i += step;
	}
};

map(function(x) { console.log('fib', x, '=', fib(x)); }, 0, 10, 1);

/******** Operators ******************************/
// Arithmetic
1 + 2 * 3 / 4 - (9 % 2);

// Comparison: >, >=, <, <=, ==, !=
// Note: uses '==' and '!=' instead of '===' and '!==' for simplicity

// Assignment: =, +=, -=, *=, /=, %=

// Unimplemented: postfix and prefix -- and ++, bitwise operators, typeof, **, instanceof, ternary, comma

// Short-circuiting logical operators
false && console.log("this does not execute");
true || console.log("this does not execute");

/******** Exceptions ******************************/
var e = "hi";
try {
	console.log(e);
	throw {error:":("};
	console.log("should not run");
} catch(e) {
	console.log("catch", e);
}
console.log(e);

/******** Scope ******************************/
// JavaScript's scope rules are weird and annoying. I ignored them and tried to
// stick to conventional scope rules. The main differences are:

// 1) Block scope
// Example:
(function() {
	var a = 0;
	var b = 0;
	console.log(a, b); // 0, 0
	if(true) {
		var b = 1;
		var c = 1;
		console.log(a, b, c); // 0, 1, 1
		a = 1;
		console.log(a, b, c); // 1, 1, 1
	}
	console.log(a, b); // 1, 0
	// JavaScript: c is availible in the whole function
	// My block scope implementation: c is only in scope inside the if block
	// c is undefined here
})();

// 2) No hoisting
// In JavaScript variables can be defined before being declared
// My implementation does not allow this
/*
// Example:
(function() {
    x = "hi";
    console.log(x); // hi
    var x;
})();
*/

// 3) No global hoisting
/*
(function(){
	// JavaScript: creates global variable
	// My implementation: causes error
	x = 1;
})();
*/

/******** Standard libraries ******************************/
// Full math standard library (up to ES5, with parts of ES6)
console.log(Math.asin(dy/dx)*180/Math.PI, 'degrees');

// Partial implementation of console
console.log('hello');
console.assert(true && (1 > 2 || 2 > 1));
```
