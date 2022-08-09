#!/usr/bin/env python
import sys, json
import numpy as np
from scipy.integrate import solve_ivp
from scipy import linalg
from numpy import cos
from numpy.linalg import solve
#from sympy import *

logfilename = "__LOG__"

class DataStruct():
	def __init__(self): 
		if len(sys.argv) != 2:
			print(f"Usage: python {sys.argv[0]} filename")
			sys.exit(0)
		fd = open(sys.argv[1], 'r')
		self.dic = json.load(fd)
		fd.close()
		self.dim = len(self.dic['x0'])

def func(t, x, data):
	dim = data.dim
	f = [ x[1], 
		-data.dic['params'][0] * x[1] - x[0]**3 
			+ data.dic['params'][1] + data.dic['params'][2] * cos(t)
	] 
	p = -3.0 * x[0] * x[0]
	k = data.dic['params'][0]
	cs = cos(t)
	dphidx = x[dim:dim*dim+dim].reshape(dim, dim).T
	dphidl = x[dim*dim:dim*dim+dim]
	dfdl = [0, cs]
	dfdx = [[0, 1], [p, -k]]
	f.extend((dfdx @ dphidx).T.flatten())
	f.extend(dfdx @ dphidl + dfdl)
	return  f

def fixed(data):
	fperiod = 2*np.pi
	duration = data.dic['period'] * fperiod
	#tspan = np.arange(0, duration, data.dic['tick'])
	x0 = data.dic['x0'] 
	dim = data.dim # number of dimension
	count = 0
	while True:
		prev = x0 
		xtemp = np.append(x0, np.eye(dim).flatten())
		data.x00 = np.append(xtemp, [0, 0])
		#print(data.x00)
		x = solve_ivp(func, (0, duration), data.x00, 
			#method = 'DOP853', 
			method = 'RK45', 
			rtol = 1e-8,
			args = (data,) )	# pass a singleton
		
		vec = x.y[:,-1]		# copy last column
		xs = vec[:dim]		# extract x
		b = prev - xs
		dphidx = vec[dim:dim*dim+dim].reshape(dim, dim).T
		A = dphidx - np.eye(dim)
		dx = solve(A, b)		# solve Ax = b
		delta = np.linalg.norm(dx, ord=2) 
		if delta < data.dic['errors']:
			break
		if delta > data.dic['explosion']:
			print("exploded.")
			sys.exit(-1)
		if count >= data.dic['ite_max']:
			print(f"over {count} times iteration")
			sys.exit(-1)
		x0 = prev + dx
		count += 1
	data.dphidx = dphidx
	data.dic['x0'] = x0
	return count

def main():
	data = DataStruct()
	with open(logfilename, mode = 'w') as logfile:
		while True:
			iteration = fixed(data)
			l = linalg.eig(data.dphidx.T)[0]
			comp = True if abs(l[0].imag) < 1e-8 else False
			str = "{0:2d} ".format(iteration)
			for i in range(len(data.dic['params'])):
				str += "{0: .5f} ".format(data.dic['params'][i])
			for i in range(len(data.dic['x0'])):
				str += "{0: .8f} ".format(data.dic['x0'][i])
			if comp == True: 
				str += "R "
				str += "{0: .7f} {1: .7f}".format(l[0].real, l[1].real)
			else:
				str += "C "
				str += "{0: .7f} {1: .7f}".format(l[0].real, l[0].imag)
			print(str)
			logfile.write(str + "\n")
			pos = data.dic['increment_param']
			data.dic['params'][pos] += data.dic['dparams'][pos]

if __name__ == '__main__':
	main()
