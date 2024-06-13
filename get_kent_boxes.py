from minimal.kent_distribution import *
from numpy.random import seed, uniform, randint	
import warnings
import sys
import json 
import pdb
from matplotlib import pyplot as plt

from skimage.io import imread

class Rotation:
    @staticmethod
    def Rx(alpha):
        return asarray([[1, 0, 0], [0, cos(alpha), -sin(alpha)], [0, sin(alpha), cos(alpha)]])

    @staticmethod
    def Ry(beta):
        return asarray([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])

    @staticmethod
    def Rz(gamma):
        return asarray([[cos(gamma), -sin(gamma), 0], [sin(gamma), cos(gamma), 0], [0, 0, 1]])

def projectEquirectangular2Sphere(u, w, h):
   #NOTE: phi and theta differ from usual definition
   theta = u[:,1] * (pi/float(h))
   phi = u[:,0] * (2.*pi/float(w))
   return vstack([cos(theta), sin(theta)*cos(phi), sin(theta)*sin(phi)]).T

def projectSphere2Equirectangular(x, w, h):
   #NOTE: phi and theta differ from usual definition
   theta = squeeze(asarray(arccos(clip(x[:,0],-1,1))))
   phi = squeeze(asarray(arctan2(x[:,2],x[:,1])))
   phi[phi < 0] += 2*pi 
   return vstack([phi * float(w)/(2*pi), theta * float(h)/(pi)])

def createSphere(I):
	h, w = I.shape #960, 1920
	v, u = mgrid[0:h:1, 0:w:1]
	#print(u.max(), v.max()) # u in [0,w), v in [0,h)]
	X = projectEquirectangular2Sphere(vstack((u.reshape(-1),v.reshape(-1))).T, w, h)
	return X, I.reshape(-1)


def plotSphere(X, c):
	x, y, z = X.T
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.set_box_aspect([1,1,1])
	ax.scatter(x, y, z, c=c)	
	plt.show()

def selectAnnotation(annotations, class_name=None):
	idx = 0
	rnd = randint(0,len(annotations['boxes']))
	for ann in annotations['boxes']:
		if class_name and ann[6] == class_name: return ann
		elif class_name is None:
			if idx == rnd: return ann
			else: idx+=1 

def sampleFromAnnotation(annotation, shape):
	h, w = shape
	data_x, data_y, _, _, data_fov_h, data_fov_v, label = annotation
	phi00 = (data_x - w / 2.) * ((2. * pi) / w)
	theta00 = -(data_y - h / 2.) * (pi / h)
	a_lat = deg2rad(data_fov_v)
	a_long = deg2rad(data_fov_h)
	r = 11
	d_lat = r / (2 * tan(a_lat / 2))
	d_long = r / (2 * tan(a_long / 2))
	p = []
	for i in range(-(r - 1) // 2, (r + 1) // 2):
		for j in range(-(r - 1) // 2, (r + 1) // 2):
			p += [asarray([i * d_lat / d_long, j, d_lat])]
	#print('.', (p[1]-p[0])	/(linalg.norm(p[1]-p[0]))) #  [0,1,0]
	#print('.', (p[r]-p[0])	/(linalg.norm(p[r]-p[0]))) #  [1,0,0]
	R = dot(Rotation.Ry(phi00), Rotation.Rx(theta00))
	p = asarray([dot(R, (p[ij] / norm(p[ij]))) for ij in range(r * r)])

	phi = asarray([arctan2(p[ij][0], p[ij][2]) for ij in range(r * r)])
	theta = asarray([arcsin(p[ij][1]) for ij in range(r * r)])
	u = (phi / (2 * pi) + 1. / 2.) * w
	v = h - (-theta / pi + 1. / 2.) * h
	return projectEquirectangular2Sphere(vstack((u,v)).T, w, h) 

if __name__ == '__main__':

	I = imread('7fB0x.jpg', as_gray=True)
	X, C = createSphere(I)
	with open('7fB0x.json') as file: A = json.load(file)

	for annotation in A['boxes']:

		annotation = selectAnnotation(A)#, 'fireplace')

		#pdb.set_trace()
		data_x, data_y, _, _, data_fov_h, data_fov_v, label = annotation
		
		phi, theta = 2*pi*data_x/I.shape[1], pi*data_y/I.shape[0]
		beta = deg2rad(data_fov_v)/deg2rad(data_fov_h)
		beta = deg2rad(tan(data_fov_v / 2))/deg2rad(tan(data_fov_h / 2))
		psi = 0
		#print(theta, phi, psi, '?', beta)
		xbar = asarray([cos(theta), sin(theta)*cos(phi), sin(theta)*sin(phi)])

		Xs = sampleFromAnnotation(annotation, I.shape)	
		k = kent_me(Xs)
		P = k.pdf(X, normalize=False)
		print(k.theta, k.phi, k.psi, k.kappa, k.beta) # theta, phi, psi, kappa, beta
		#pdb.set_trace()
		
		"""plt.figure()
		plt.imshow(P.reshape(I.shape), cmap='Oranges')
		plt.imshow(C.reshape(I.shape), cmap='gray', alpha=.7)
		plt.scatter(*projectSphere2Equirectangular(Xs,  I.shape[1], I.shape[0]), s=1,c='blue',alpha=.1)

		plt.show()"""
	

