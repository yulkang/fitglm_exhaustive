function s=dec2bin(d,n,varargin)
%DEC2BIN Convert decimal integer to a binary string.
%   DEC2BIN(D) returns the binary representation of D as a string.
%   D must be a non-negative integer smaller than 2^52.
%
%   DEC2BIN(D,N) produces a binary representation with at least
%   N bits.
%
%   Example
%      dec2bin(23) returns '10111'
%
%   See also BIN2DEC, DEC2HEX, DEC2BASE.

%   Copyright 1984-2009 The MathWorks, Inc.

%
% Input checking
%
if nargin<1
    narginchk(1,2);
end
S = varargin2S(varargin, {
    'chunk', 4
    });
if isempty(d)
    s = '';
    return;
end

if ~(isnumeric(d) || islogical(d) || ischar(d))
    error(message('MATLAB:dec2bin:InvalidDecimalArg'));
end
d = d(:); % Make sure d is a column vector.
if any(d < 0) || any(~isfinite(d))
    error(message('MATLAB:dec2bin:MustBeNonNegativeFinite'));
end

if ~isreal(d)
    error(message('MATLAB:dec2bin:MustBeReal'));
end
    
d = double(d);

if nargin<2
    n=1; % Need at least one digit even for 0.
else
    if ~(isnumeric(n) || ischar(n)) || ~isscalar(n) || n<0
        error(message('MATLAB:dec2bin:InvalidBitArg'));
    end
    n = double(n);
    n = round(n); % Make sure n is an integer.
end;

%
% Actual algorithm
%
[f,e]=log2(max(d)); % How many digits do we need to represent the numbers?
s=char(rem(floor(d*pow2(1-max(n,e):0)),2)+'0');
