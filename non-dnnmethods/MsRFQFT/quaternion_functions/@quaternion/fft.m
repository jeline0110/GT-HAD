function Y = fft(X)
% FFT Discrete Quaternion Fourier transform.
% (Quaternion overloading of standard Matlab function, but only one parameter.)
% (The parameters N and dim of the standard function are not yet implemented.)
% 
% This function implements a default quaternion FFT. See the related function
% QFFT, which implements transforms with left or right exponentials and a
% user-specified axis.

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

% Implementation note: keeping QFFT separate means that the quaternion-specific
% parameters (axis and left/right) are kept separate from the Matlab standard
% parameters (N and dim) which might be added here at a later date.

error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout)) 

Y = qfft(X, dft_axis(isreal(X)), 'L');

% $Id: fft.m,v 1.3 2009/02/08 18:35:21 sangwine Exp $

