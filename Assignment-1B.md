# **Assignment-1B**

## What are Channels and Kernels (according to EVA)?

**Filers / Kernels**

A very simple definition may be - The sets of weights which is convolved with the input, is called as Filter/kernel. To extract features from an input image, lets assume a focused beam of light(for ex. a Flashlight) is zooming/focusing over the top left corner of the image, and light emitting from this flashlight is focusing over an area of 3 x 3 pixels. This flashlight is sliding all over this image, to focus on other areas as well, to see the contents of the image. Here this flashlight is acting like a **filter**(or sometimes also referred as a **neuron** or a **kernel** or a **feature identifier**).
A Flashlight, is acting as a **filter** is basically a matrix of 3 x 3, or 5 x 5, or 7 x 7 dimension. 


$$
Filter / Kernel =
   \begin{vmatrix}
    1 & -1 \\
    -1 & 1 \\
   \end{vmatrix}\\
$$


The main objective of the filter is to extract features of an image. In the first convolution layer filters are extracting low level features like edges and curves. Generally one kernel is used for detecting a unique feature. In order to classify or predict an image, we need more layers of filters, which extracts higher level features such as hands, nose, eyes etc. 

**Channels**

If we have a grey scale image, this means that we are getting data from one single sensor. If we are having a RGB i.e. colored image then we are getting data from 3 sensors. In print media, four color is standardized to cyan, magenta, yellow, and black (or key), i.e we are getting data from 4 sensors.

Hence, this means that a channel can be understood or considered as collection of same information seen from different sensors / perspective(i.e. color).

So, If we see how the kernel (i.e. 5x5x3, or flashlight as mention above) moves in a CNN, then it moves in XY direction and not in the channel direction. So, we are trying to learn features in XY direction from all the channels together.  In other words 3 different 2D filters of size 5x5 can be concatenated and are moving in XY direction. And there are as many 2D filters as number of input channels in the image.

## Why should we only (well mostly) use 3x3 Kernels?

 3 x 3 filter is used as a defacto convolution operation in a neural network, because - 
1. Advantage of using a smaller size filters such as 3 x 3 filter is that, they are faster in terms of computational cost over a 5 x 5 or a 7 x 7. And can significantly reduces the number of features to be learnt in a neural network. For ex : Two 3 x 3 kernels will result in an image size reduction by 4. Which is same as one 5 x 5 layer. But two 3 x 3 kernel will result in 1 weights while a 5 x 5 kernel will result in 25 weights. Therefore a 3 x 3 kernel is  computationally efficient.
2.  With more layers, means network will learns complex and more non-linear features.
3. 3 x 3 filter is faster then why we don't use smaller even sized (2 x 2) kernel, because they don't produce a centered results.
4. In an odd-sized filter, all the previous layer pixels would be symmetrically distributed around its output pixel. Without this symmetry, will lead to distortions across the layers which occurs due to using an even sized kernel.

## How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations).





Layer	--	1	199	x	199	|	3	x	3	Kernel	--	197	x	197
Layer	--	2	197	x	197	|	3	x	3	Kernel	--	195	x	195
Layer	--	3	195	x	195	|	3	x	3	Kernel	--	193	x	193
Layer	--	4	193	x	193	|	3	x	3	Kernel	--	191	x	191
Layer	--	5	191	x	191	|	3	x	3	Kernel	--	189	x	189
Layer	--	6	189	x	189	|	3	x	3	Kernel	--	187	x	187
Layer	--	7	187	x	187	|	3	x	3	Kernel	--	185	x	185
Layer	--	8	185	x	185	|	3	x	3	Kernel	--	183	x	183
Layer	--	9	183	x	183	|	3	x	3	Kernel	--	181	x	181
Layer	--	10	181	x	181	|	3	x	3	Kernel	--	179	x	179
Layer	--	11	179	x	179	|	3	x	3	Kernel	--	177	x	177
Layer	--	12	177	x	177	|	3	x	3	Kernel	--	175	x	175
Layer	--	13	175	x	175	|	3	x	3	Kernel	--	173	x	173
Layer	--	14	173	x	173	|	3	x	3	Kernel	--	171	x	171
Layer	--	15	171	x	171	|	3	x	3	Kernel	--	169	x	169
Layer	--	16	169	x	169	|	3	x	3	Kernel	--	167	x	167
Layer	--	17	167	x	167	|	3	x	3	Kernel	--	165	x	165
Layer	--	18	165	x	165	|	3	x	3	Kernel	--	163	x	163
Layer	--	19	163	x	163	|	3	x	3	Kernel	--	161	x	161
Layer	--	20	161	x	161	|	3	x	3	Kernel	--	159	x	159
Layer	--	21	159	x	159	|	3	x	3	Kernel	--	157	x	157
Layer	--	22	157	x	157	|	3	x	3	Kernel	--	155	x	155
Layer	--	23	155	x	155	|	3	x	3	Kernel	--	153	x	153
Layer	--	24	153	x	153	|	3	x	3	Kernel	--	151	x	151
Layer	--	25	151	x	151	|	3	x	3	Kernel	--	149	x	149
Layer	--	26	149	x	149	|	3	x	3	Kernel	--	147	x	147
Layer	--	27	147	x	147	|	3	x	3	Kernel	--	145	x	145
Layer	--	28	145	x	145	|	3	x	3	Kernel	--	143	x	143
Layer	--	29	143	x	143	|	3	x	3	Kernel	--	141	x	141
Layer	--	30	141	x	141	|	3	x	3	Kernel	--	139	x	139
Layer	--	31	139	x	139	|	3	x	3	Kernel	--	137	x	137
Layer	--	32	137	x	137	|	3	x	3	Kernel	--	135	x	135
Layer	--	33	135	x	135	|	3	x	3	Kernel	--	133	x	133
Layer	--	34	133	x	133	|	3	x	3	Kernel	--	131	x	131
Layer	--	35	131	x	131	|	3	x	3	Kernel	--	129	x	129
Layer	--	36	129	x	129	|	3	x	3	Kernel	--	127	x	127
Layer	--	37	127	x	127	|	3	x	3	Kernel	--	125	x	125
Layer	--	38	125	x	125	|	3	x	3	Kernel	--	123	x	123
Layer	--	39	123	x	123	|	3	x	3	Kernel	--	121	x	121
Layer	--	40	121	x	121	|	3	x	3	Kernel	--	119	x	119
Layer	--	41	119	x	119	|	3	x	3	Kernel	--	117	x	117
Layer	--	42	117	x	117	|	3	x	3	Kernel	--	115	x	115
Layer	--	43	115	x	115	|	3	x	3	Kernel	--	113	x	113
Layer	--	44	113	x	113	|	3	x	3	Kernel	--	111	x	111
Layer	--	45	111	x	111	|	3	x	3	Kernel	--	109	x	109
Layer	--	46	109	x	109	|	3	x	3	Kernel	--	107	x	107
Layer	--	47	107	x	107	|	3	x	3	Kernel	--	105	x	105
Layer	--	48	105	x	105	|	3	x	3	Kernel	--	103	x	103
Layer	--	49	103	x	103	|	3	x	3	Kernel	--	101	x	101
Layer	--	50	101	x	101	|	3	x	3	Kernel	--	99	x	99
Layer	--	51	99	x	99	|	3	x	3	Kernel	--	97	x	97
Layer	--	52	97	x	97	|	3	x	3	Kernel	--	95	x	95
Layer	--	53	95	x	95	|	3	x	3	Kernel	--	93	x	93
Layer	--	54	93	x	93	|	3	x	3	Kernel	--	91	x	91
Layer	--	55	91	x	91	|	3	x	3	Kernel	--	89	x	89
Layer	--	56	89	x	89	|	3	x	3	Kernel	--	87	x	87
Layer	--	57	87	x	87	|	3	x	3	Kernel	--	85	x	85
Layer	--	58	85	x	85	|	3	x	3	Kernel	--	83	x	83
Layer	--	59	83	x	83	|	3	x	3	Kernel	--	81	x	81
Layer	--	60	81	x	81	|	3	x	3	Kernel	--	79	x	79
Layer	--	61	79	x	79	|	3	x	3	Kernel	--	77	x	77
Layer	--	62	77	x	77	|	3	x	3	Kernel	--	75	x	75
Layer	--	63	75	x	75	|	3	x	3	Kernel	--	73	x	73
Layer	--	64	73	x	73	|	3	x	3	Kernel	--	71	x	71
Layer	--	65	71	x	71	|	3	x	3	Kernel	--	69	x	69
Layer	--	66	69	x	69	|	3	x	3	Kernel	--	67	x	67
Layer	--	67	67	x	67	|	3	x	3	Kernel	--	65	x	65
Layer	--	68	65	x	65	|	3	x	3	Kernel	--	63	x	63
Layer	--	69	63	x	63	|	3	x	3	Kernel	--	61	x	61
Layer	--	70	61	x	61	|	3	x	3	Kernel	--	59	x	59
Layer	--	71	59	x	59	|	3	x	3	Kernel	--	57	x	57
Layer	--	72	57	x	57	|	3	x	3	Kernel	--	55	x	55
Layer	--	73	55	x	55	|	3	x	3	Kernel	--	53	x	53
Layer	--	74	53	x	53	|	3	x	3	Kernel	--	51	x	51
Layer	--	75	51	x	51	|	3	x	3	Kernel	--	49	x	49
Layer	--	76	49	x	49	|	3	x	3	Kernel	--	47	x	47
Layer	--	77	47	x	47	|	3	x	3	Kernel	--	45	x	45
Layer	--	78	45	x	45	|	3	x	3	Kernel	--	43	x	43
Layer	--	79	43	x	43	|	3	x	3	Kernel	--	41	x	41
Layer	--	80	41	x	41	|	3	x	3	Kernel	--	39	x	39
Layer	--	81	39	x	39	|	3	x	3	Kernel	--	37	x	37
Layer	--	82	37	x	37	|	3	x	3	Kernel	--	35	x	35
Layer	--	83	35	x	35	|	3	x	3	Kernel	--	33	x	33
Layer	--	84	33	x	33	|	3	x	3	Kernel	--	31	x	31
Layer	--	85	31	x	31	|	3	x	3	Kernel	--	29	x	29
Layer	--	86	29	x	29	|	3	x	3	Kernel	--	27	x	27
Layer	--	87	27	x	27	|	3	x	3	Kernel	--	25	x	25
Layer	--	88	25	x	25	|	3	x	3	Kernel	--	23	x	23
Layer	--	89	23	x	23	|	3	x	3	Kernel	--	21	x	21
Layer	--	90	21	x	21	|	3	x	3	Kernel	--	19	x	19
Layer	--	91	19	x	19	|	3	x	3	Kernel	--	17	x	17
Layer	--	92	17	x	17	|	3	x	3	Kernel	--	15	x	15
Layer	--	93	15	x	15	|	3	x	3	Kernel	--	13	x	13
Layer	--	94	13	x	13	|	3	x	3	Kernel	--	11	x	11
Layer	--	95	11	x	11	|	3	x	3	Kernel	--	9	x	9
Layer	--	96	9	x	9	|	3	x	3	Kernel	--	7	x	7
Layer	--	97	7	x	7	|	3	x	3	Kernel	--	5	x	5
Layer	--	98	5	x	5	|	3	x	3	Kernel	--	3	x	3
Layer	--	99	3	x	3	|	3	x	3	Kernel	--	1	x	1

