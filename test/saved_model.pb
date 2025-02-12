��8
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48��.
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
�
v/output_C/biasVarHandleOp*
_output_shapes
: * 

debug_namev/output_C/bias/*
dtype0*
shape:* 
shared_namev/output_C/bias
o
#v/output_C/bias/Read/ReadVariableOpReadVariableOpv/output_C/bias*
_output_shapes
:*
dtype0
�
m/output_C/biasVarHandleOp*
_output_shapes
: * 

debug_namem/output_C/bias/*
dtype0*
shape:* 
shared_namem/output_C/bias
o
#m/output_C/bias/Read/ReadVariableOpReadVariableOpm/output_C/bias*
_output_shapes
:*
dtype0
�
v/output_C/kernelVarHandleOp*
_output_shapes
: *"

debug_namev/output_C/kernel/*
dtype0*
shape:A*"
shared_namev/output_C/kernel

%v/output_C/kernel/Read/ReadVariableOpReadVariableOpv/output_C/kernel*&
_output_shapes
:A*
dtype0
�
m/output_C/kernelVarHandleOp*
_output_shapes
: *"

debug_namem/output_C/kernel/*
dtype0*
shape:A*"
shared_namem/output_C/kernel

%m/output_C/kernel/Read/ReadVariableOpReadVariableOpm/output_C/kernel*&
_output_shapes
:A*
dtype0
�
v/conv_hid5_5x5/biasVarHandleOp*
_output_shapes
: *%

debug_namev/conv_hid5_5x5/bias/*
dtype0*
shape:*%
shared_namev/conv_hid5_5x5/bias
y
(v/conv_hid5_5x5/bias/Read/ReadVariableOpReadVariableOpv/conv_hid5_5x5/bias*
_output_shapes
:*
dtype0
�
m/conv_hid5_5x5/biasVarHandleOp*
_output_shapes
: *%

debug_namem/conv_hid5_5x5/bias/*
dtype0*
shape:*%
shared_namem/conv_hid5_5x5/bias
y
(m/conv_hid5_5x5/bias/Read/ReadVariableOpReadVariableOpm/conv_hid5_5x5/bias*
_output_shapes
:*
dtype0
�
v/conv_hid5_5x5/kernelVarHandleOp*
_output_shapes
: *'

debug_namev/conv_hid5_5x5/kernel/*
dtype0*
shape:A*'
shared_namev/conv_hid5_5x5/kernel
�
*v/conv_hid5_5x5/kernel/Read/ReadVariableOpReadVariableOpv/conv_hid5_5x5/kernel*&
_output_shapes
:A*
dtype0
�
m/conv_hid5_5x5/kernelVarHandleOp*
_output_shapes
: *'

debug_namem/conv_hid5_5x5/kernel/*
dtype0*
shape:A*'
shared_namem/conv_hid5_5x5/kernel
�
*m/conv_hid5_5x5/kernel/Read/ReadVariableOpReadVariableOpm/conv_hid5_5x5/kernel*&
_output_shapes
:A*
dtype0
�
v/conv_hid4_4x4/biasVarHandleOp*
_output_shapes
: *%

debug_namev/conv_hid4_4x4/bias/*
dtype0*
shape:
*%
shared_namev/conv_hid4_4x4/bias
y
(v/conv_hid4_4x4/bias/Read/ReadVariableOpReadVariableOpv/conv_hid4_4x4/bias*
_output_shapes
:
*
dtype0
�
m/conv_hid4_4x4/biasVarHandleOp*
_output_shapes
: *%

debug_namem/conv_hid4_4x4/bias/*
dtype0*
shape:
*%
shared_namem/conv_hid4_4x4/bias
y
(m/conv_hid4_4x4/bias/Read/ReadVariableOpReadVariableOpm/conv_hid4_4x4/bias*
_output_shapes
:
*
dtype0
�
v/conv_hid4_4x4/kernelVarHandleOp*
_output_shapes
: *'

debug_namev/conv_hid4_4x4/kernel/*
dtype0*
shape:A
*'
shared_namev/conv_hid4_4x4/kernel
�
*v/conv_hid4_4x4/kernel/Read/ReadVariableOpReadVariableOpv/conv_hid4_4x4/kernel*&
_output_shapes
:A
*
dtype0
�
m/conv_hid4_4x4/kernelVarHandleOp*
_output_shapes
: *'

debug_namem/conv_hid4_4x4/kernel/*
dtype0*
shape:A
*'
shared_namem/conv_hid4_4x4/kernel
�
*m/conv_hid4_4x4/kernel/Read/ReadVariableOpReadVariableOpm/conv_hid4_4x4/kernel*&
_output_shapes
:A
*
dtype0
�
v/conv_hid4_3x3/biasVarHandleOp*
_output_shapes
: *%

debug_namev/conv_hid4_3x3/bias/*
dtype0*
shape:2*%
shared_namev/conv_hid4_3x3/bias
y
(v/conv_hid4_3x3/bias/Read/ReadVariableOpReadVariableOpv/conv_hid4_3x3/bias*
_output_shapes
:2*
dtype0
�
m/conv_hid4_3x3/biasVarHandleOp*
_output_shapes
: *%

debug_namem/conv_hid4_3x3/bias/*
dtype0*
shape:2*%
shared_namem/conv_hid4_3x3/bias
y
(m/conv_hid4_3x3/bias/Read/ReadVariableOpReadVariableOpm/conv_hid4_3x3/bias*
_output_shapes
:2*
dtype0
�
v/conv_hid4_3x3/kernelVarHandleOp*
_output_shapes
: *'

debug_namev/conv_hid4_3x3/kernel/*
dtype0*
shape:A2*'
shared_namev/conv_hid4_3x3/kernel
�
*v/conv_hid4_3x3/kernel/Read/ReadVariableOpReadVariableOpv/conv_hid4_3x3/kernel*&
_output_shapes
:A2*
dtype0
�
m/conv_hid4_3x3/kernelVarHandleOp*
_output_shapes
: *'

debug_namem/conv_hid4_3x3/kernel/*
dtype0*
shape:A2*'
shared_namem/conv_hid4_3x3/kernel
�
*m/conv_hid4_3x3/kernel/Read/ReadVariableOpReadVariableOpm/conv_hid4_3x3/kernel*&
_output_shapes
:A2*
dtype0
�
v/conv_hid3_5x5/biasVarHandleOp*
_output_shapes
: *%

debug_namev/conv_hid3_5x5/bias/*
dtype0*
shape:*%
shared_namev/conv_hid3_5x5/bias
y
(v/conv_hid3_5x5/bias/Read/ReadVariableOpReadVariableOpv/conv_hid3_5x5/bias*
_output_shapes
:*
dtype0
�
m/conv_hid3_5x5/biasVarHandleOp*
_output_shapes
: *%

debug_namem/conv_hid3_5x5/bias/*
dtype0*
shape:*%
shared_namem/conv_hid3_5x5/bias
y
(m/conv_hid3_5x5/bias/Read/ReadVariableOpReadVariableOpm/conv_hid3_5x5/bias*
_output_shapes
:*
dtype0
�
v/conv_hid3_5x5/kernelVarHandleOp*
_output_shapes
: *'

debug_namev/conv_hid3_5x5/kernel/*
dtype0*
shape:A*'
shared_namev/conv_hid3_5x5/kernel
�
*v/conv_hid3_5x5/kernel/Read/ReadVariableOpReadVariableOpv/conv_hid3_5x5/kernel*&
_output_shapes
:A*
dtype0
�
m/conv_hid3_5x5/kernelVarHandleOp*
_output_shapes
: *'

debug_namem/conv_hid3_5x5/kernel/*
dtype0*
shape:A*'
shared_namem/conv_hid3_5x5/kernel
�
*m/conv_hid3_5x5/kernel/Read/ReadVariableOpReadVariableOpm/conv_hid3_5x5/kernel*&
_output_shapes
:A*
dtype0
�
v/conv_hid3_4x4/biasVarHandleOp*
_output_shapes
: *%

debug_namev/conv_hid3_4x4/bias/*
dtype0*
shape:
*%
shared_namev/conv_hid3_4x4/bias
y
(v/conv_hid3_4x4/bias/Read/ReadVariableOpReadVariableOpv/conv_hid3_4x4/bias*
_output_shapes
:
*
dtype0
�
m/conv_hid3_4x4/biasVarHandleOp*
_output_shapes
: *%

debug_namem/conv_hid3_4x4/bias/*
dtype0*
shape:
*%
shared_namem/conv_hid3_4x4/bias
y
(m/conv_hid3_4x4/bias/Read/ReadVariableOpReadVariableOpm/conv_hid3_4x4/bias*
_output_shapes
:
*
dtype0
�
v/conv_hid3_4x4/kernelVarHandleOp*
_output_shapes
: *'

debug_namev/conv_hid3_4x4/kernel/*
dtype0*
shape:A
*'
shared_namev/conv_hid3_4x4/kernel
�
*v/conv_hid3_4x4/kernel/Read/ReadVariableOpReadVariableOpv/conv_hid3_4x4/kernel*&
_output_shapes
:A
*
dtype0
�
m/conv_hid3_4x4/kernelVarHandleOp*
_output_shapes
: *'

debug_namem/conv_hid3_4x4/kernel/*
dtype0*
shape:A
*'
shared_namem/conv_hid3_4x4/kernel
�
*m/conv_hid3_4x4/kernel/Read/ReadVariableOpReadVariableOpm/conv_hid3_4x4/kernel*&
_output_shapes
:A
*
dtype0
�
v/conv_hid3_3x3/biasVarHandleOp*
_output_shapes
: *%

debug_namev/conv_hid3_3x3/bias/*
dtype0*
shape:2*%
shared_namev/conv_hid3_3x3/bias
y
(v/conv_hid3_3x3/bias/Read/ReadVariableOpReadVariableOpv/conv_hid3_3x3/bias*
_output_shapes
:2*
dtype0
�
m/conv_hid3_3x3/biasVarHandleOp*
_output_shapes
: *%

debug_namem/conv_hid3_3x3/bias/*
dtype0*
shape:2*%
shared_namem/conv_hid3_3x3/bias
y
(m/conv_hid3_3x3/bias/Read/ReadVariableOpReadVariableOpm/conv_hid3_3x3/bias*
_output_shapes
:2*
dtype0
�
v/conv_hid3_3x3/kernelVarHandleOp*
_output_shapes
: *'

debug_namev/conv_hid3_3x3/kernel/*
dtype0*
shape:A2*'
shared_namev/conv_hid3_3x3/kernel
�
*v/conv_hid3_3x3/kernel/Read/ReadVariableOpReadVariableOpv/conv_hid3_3x3/kernel*&
_output_shapes
:A2*
dtype0
�
m/conv_hid3_3x3/kernelVarHandleOp*
_output_shapes
: *'

debug_namem/conv_hid3_3x3/kernel/*
dtype0*
shape:A2*'
shared_namem/conv_hid3_3x3/kernel
�
*m/conv_hid3_3x3/kernel/Read/ReadVariableOpReadVariableOpm/conv_hid3_3x3/kernel*&
_output_shapes
:A2*
dtype0
�
v/conv_hid2_5x5/biasVarHandleOp*
_output_shapes
: *%

debug_namev/conv_hid2_5x5/bias/*
dtype0*
shape:*%
shared_namev/conv_hid2_5x5/bias
y
(v/conv_hid2_5x5/bias/Read/ReadVariableOpReadVariableOpv/conv_hid2_5x5/bias*
_output_shapes
:*
dtype0
�
m/conv_hid2_5x5/biasVarHandleOp*
_output_shapes
: *%

debug_namem/conv_hid2_5x5/bias/*
dtype0*
shape:*%
shared_namem/conv_hid2_5x5/bias
y
(m/conv_hid2_5x5/bias/Read/ReadVariableOpReadVariableOpm/conv_hid2_5x5/bias*
_output_shapes
:*
dtype0
�
v/conv_hid2_5x5/kernelVarHandleOp*
_output_shapes
: *'

debug_namev/conv_hid2_5x5/kernel/*
dtype0*
shape:A*'
shared_namev/conv_hid2_5x5/kernel
�
*v/conv_hid2_5x5/kernel/Read/ReadVariableOpReadVariableOpv/conv_hid2_5x5/kernel*&
_output_shapes
:A*
dtype0
�
m/conv_hid2_5x5/kernelVarHandleOp*
_output_shapes
: *'

debug_namem/conv_hid2_5x5/kernel/*
dtype0*
shape:A*'
shared_namem/conv_hid2_5x5/kernel
�
*m/conv_hid2_5x5/kernel/Read/ReadVariableOpReadVariableOpm/conv_hid2_5x5/kernel*&
_output_shapes
:A*
dtype0
�
v/conv_hid2_4x4/biasVarHandleOp*
_output_shapes
: *%

debug_namev/conv_hid2_4x4/bias/*
dtype0*
shape:
*%
shared_namev/conv_hid2_4x4/bias
y
(v/conv_hid2_4x4/bias/Read/ReadVariableOpReadVariableOpv/conv_hid2_4x4/bias*
_output_shapes
:
*
dtype0
�
m/conv_hid2_4x4/biasVarHandleOp*
_output_shapes
: *%

debug_namem/conv_hid2_4x4/bias/*
dtype0*
shape:
*%
shared_namem/conv_hid2_4x4/bias
y
(m/conv_hid2_4x4/bias/Read/ReadVariableOpReadVariableOpm/conv_hid2_4x4/bias*
_output_shapes
:
*
dtype0
�
v/conv_hid2_4x4/kernelVarHandleOp*
_output_shapes
: *'

debug_namev/conv_hid2_4x4/kernel/*
dtype0*
shape:A
*'
shared_namev/conv_hid2_4x4/kernel
�
*v/conv_hid2_4x4/kernel/Read/ReadVariableOpReadVariableOpv/conv_hid2_4x4/kernel*&
_output_shapes
:A
*
dtype0
�
m/conv_hid2_4x4/kernelVarHandleOp*
_output_shapes
: *'

debug_namem/conv_hid2_4x4/kernel/*
dtype0*
shape:A
*'
shared_namem/conv_hid2_4x4/kernel
�
*m/conv_hid2_4x4/kernel/Read/ReadVariableOpReadVariableOpm/conv_hid2_4x4/kernel*&
_output_shapes
:A
*
dtype0
�
v/conv_hid2_3x3/biasVarHandleOp*
_output_shapes
: *%

debug_namev/conv_hid2_3x3/bias/*
dtype0*
shape:2*%
shared_namev/conv_hid2_3x3/bias
y
(v/conv_hid2_3x3/bias/Read/ReadVariableOpReadVariableOpv/conv_hid2_3x3/bias*
_output_shapes
:2*
dtype0
�
m/conv_hid2_3x3/biasVarHandleOp*
_output_shapes
: *%

debug_namem/conv_hid2_3x3/bias/*
dtype0*
shape:2*%
shared_namem/conv_hid2_3x3/bias
y
(m/conv_hid2_3x3/bias/Read/ReadVariableOpReadVariableOpm/conv_hid2_3x3/bias*
_output_shapes
:2*
dtype0
�
v/conv_hid2_3x3/kernelVarHandleOp*
_output_shapes
: *'

debug_namev/conv_hid2_3x3/kernel/*
dtype0*
shape:A2*'
shared_namev/conv_hid2_3x3/kernel
�
*v/conv_hid2_3x3/kernel/Read/ReadVariableOpReadVariableOpv/conv_hid2_3x3/kernel*&
_output_shapes
:A2*
dtype0
�
m/conv_hid2_3x3/kernelVarHandleOp*
_output_shapes
: *'

debug_namem/conv_hid2_3x3/kernel/*
dtype0*
shape:A2*'
shared_namem/conv_hid2_3x3/kernel
�
*m/conv_hid2_3x3/kernel/Read/ReadVariableOpReadVariableOpm/conv_hid2_3x3/kernel*&
_output_shapes
:A2*
dtype0
�
v/conv_hid1_5x5/biasVarHandleOp*
_output_shapes
: *%

debug_namev/conv_hid1_5x5/bias/*
dtype0*
shape:*%
shared_namev/conv_hid1_5x5/bias
y
(v/conv_hid1_5x5/bias/Read/ReadVariableOpReadVariableOpv/conv_hid1_5x5/bias*
_output_shapes
:*
dtype0
�
m/conv_hid1_5x5/biasVarHandleOp*
_output_shapes
: *%

debug_namem/conv_hid1_5x5/bias/*
dtype0*
shape:*%
shared_namem/conv_hid1_5x5/bias
y
(m/conv_hid1_5x5/bias/Read/ReadVariableOpReadVariableOpm/conv_hid1_5x5/bias*
_output_shapes
:*
dtype0
�
v/conv_hid1_5x5/kernelVarHandleOp*
_output_shapes
: *'

debug_namev/conv_hid1_5x5/kernel/*
dtype0*
shape:A*'
shared_namev/conv_hid1_5x5/kernel
�
*v/conv_hid1_5x5/kernel/Read/ReadVariableOpReadVariableOpv/conv_hid1_5x5/kernel*&
_output_shapes
:A*
dtype0
�
m/conv_hid1_5x5/kernelVarHandleOp*
_output_shapes
: *'

debug_namem/conv_hid1_5x5/kernel/*
dtype0*
shape:A*'
shared_namem/conv_hid1_5x5/kernel
�
*m/conv_hid1_5x5/kernel/Read/ReadVariableOpReadVariableOpm/conv_hid1_5x5/kernel*&
_output_shapes
:A*
dtype0
�
v/conv_hid1_4x4/biasVarHandleOp*
_output_shapes
: *%

debug_namev/conv_hid1_4x4/bias/*
dtype0*
shape:
*%
shared_namev/conv_hid1_4x4/bias
y
(v/conv_hid1_4x4/bias/Read/ReadVariableOpReadVariableOpv/conv_hid1_4x4/bias*
_output_shapes
:
*
dtype0
�
m/conv_hid1_4x4/biasVarHandleOp*
_output_shapes
: *%

debug_namem/conv_hid1_4x4/bias/*
dtype0*
shape:
*%
shared_namem/conv_hid1_4x4/bias
y
(m/conv_hid1_4x4/bias/Read/ReadVariableOpReadVariableOpm/conv_hid1_4x4/bias*
_output_shapes
:
*
dtype0
�
v/conv_hid1_4x4/kernelVarHandleOp*
_output_shapes
: *'

debug_namev/conv_hid1_4x4/kernel/*
dtype0*
shape:A
*'
shared_namev/conv_hid1_4x4/kernel
�
*v/conv_hid1_4x4/kernel/Read/ReadVariableOpReadVariableOpv/conv_hid1_4x4/kernel*&
_output_shapes
:A
*
dtype0
�
m/conv_hid1_4x4/kernelVarHandleOp*
_output_shapes
: *'

debug_namem/conv_hid1_4x4/kernel/*
dtype0*
shape:A
*'
shared_namem/conv_hid1_4x4/kernel
�
*m/conv_hid1_4x4/kernel/Read/ReadVariableOpReadVariableOpm/conv_hid1_4x4/kernel*&
_output_shapes
:A
*
dtype0
�
v/conv_hid1_3x3/biasVarHandleOp*
_output_shapes
: *%

debug_namev/conv_hid1_3x3/bias/*
dtype0*
shape:2*%
shared_namev/conv_hid1_3x3/bias
y
(v/conv_hid1_3x3/bias/Read/ReadVariableOpReadVariableOpv/conv_hid1_3x3/bias*
_output_shapes
:2*
dtype0
�
m/conv_hid1_3x3/biasVarHandleOp*
_output_shapes
: *%

debug_namem/conv_hid1_3x3/bias/*
dtype0*
shape:2*%
shared_namem/conv_hid1_3x3/bias
y
(m/conv_hid1_3x3/bias/Read/ReadVariableOpReadVariableOpm/conv_hid1_3x3/bias*
_output_shapes
:2*
dtype0
�
v/conv_hid1_3x3/kernelVarHandleOp*
_output_shapes
: *'

debug_namev/conv_hid1_3x3/kernel/*
dtype0*
shape:A2*'
shared_namev/conv_hid1_3x3/kernel
�
*v/conv_hid1_3x3/kernel/Read/ReadVariableOpReadVariableOpv/conv_hid1_3x3/kernel*&
_output_shapes
:A2*
dtype0
�
m/conv_hid1_3x3/kernelVarHandleOp*
_output_shapes
: *'

debug_namem/conv_hid1_3x3/kernel/*
dtype0*
shape:A2*'
shared_namem/conv_hid1_3x3/kernel
�
*m/conv_hid1_3x3/kernel/Read/ReadVariableOpReadVariableOpm/conv_hid1_3x3/kernel*&
_output_shapes
:A2*
dtype0
�
v/conv_hid0_5x5/biasVarHandleOp*
_output_shapes
: *%

debug_namev/conv_hid0_5x5/bias/*
dtype0*
shape:*%
shared_namev/conv_hid0_5x5/bias
y
(v/conv_hid0_5x5/bias/Read/ReadVariableOpReadVariableOpv/conv_hid0_5x5/bias*
_output_shapes
:*
dtype0
�
m/conv_hid0_5x5/biasVarHandleOp*
_output_shapes
: *%

debug_namem/conv_hid0_5x5/bias/*
dtype0*
shape:*%
shared_namem/conv_hid0_5x5/bias
y
(m/conv_hid0_5x5/bias/Read/ReadVariableOpReadVariableOpm/conv_hid0_5x5/bias*
_output_shapes
:*
dtype0
�
v/conv_hid0_5x5/kernelVarHandleOp*
_output_shapes
: *'

debug_namev/conv_hid0_5x5/kernel/*
dtype0*
shape:D*'
shared_namev/conv_hid0_5x5/kernel
�
*v/conv_hid0_5x5/kernel/Read/ReadVariableOpReadVariableOpv/conv_hid0_5x5/kernel*&
_output_shapes
:D*
dtype0
�
m/conv_hid0_5x5/kernelVarHandleOp*
_output_shapes
: *'

debug_namem/conv_hid0_5x5/kernel/*
dtype0*
shape:D*'
shared_namem/conv_hid0_5x5/kernel
�
*m/conv_hid0_5x5/kernel/Read/ReadVariableOpReadVariableOpm/conv_hid0_5x5/kernel*&
_output_shapes
:D*
dtype0
�
v/conv_hid0_4x4/biasVarHandleOp*
_output_shapes
: *%

debug_namev/conv_hid0_4x4/bias/*
dtype0*
shape:
*%
shared_namev/conv_hid0_4x4/bias
y
(v/conv_hid0_4x4/bias/Read/ReadVariableOpReadVariableOpv/conv_hid0_4x4/bias*
_output_shapes
:
*
dtype0
�
m/conv_hid0_4x4/biasVarHandleOp*
_output_shapes
: *%

debug_namem/conv_hid0_4x4/bias/*
dtype0*
shape:
*%
shared_namem/conv_hid0_4x4/bias
y
(m/conv_hid0_4x4/bias/Read/ReadVariableOpReadVariableOpm/conv_hid0_4x4/bias*
_output_shapes
:
*
dtype0
�
v/conv_hid0_4x4/kernelVarHandleOp*
_output_shapes
: *'

debug_namev/conv_hid0_4x4/kernel/*
dtype0*
shape:D
*'
shared_namev/conv_hid0_4x4/kernel
�
*v/conv_hid0_4x4/kernel/Read/ReadVariableOpReadVariableOpv/conv_hid0_4x4/kernel*&
_output_shapes
:D
*
dtype0
�
m/conv_hid0_4x4/kernelVarHandleOp*
_output_shapes
: *'

debug_namem/conv_hid0_4x4/kernel/*
dtype0*
shape:D
*'
shared_namem/conv_hid0_4x4/kernel
�
*m/conv_hid0_4x4/kernel/Read/ReadVariableOpReadVariableOpm/conv_hid0_4x4/kernel*&
_output_shapes
:D
*
dtype0
�
v/conv_hid0_3x3/biasVarHandleOp*
_output_shapes
: *%

debug_namev/conv_hid0_3x3/bias/*
dtype0*
shape:2*%
shared_namev/conv_hid0_3x3/bias
y
(v/conv_hid0_3x3/bias/Read/ReadVariableOpReadVariableOpv/conv_hid0_3x3/bias*
_output_shapes
:2*
dtype0
�
m/conv_hid0_3x3/biasVarHandleOp*
_output_shapes
: *%

debug_namem/conv_hid0_3x3/bias/*
dtype0*
shape:2*%
shared_namem/conv_hid0_3x3/bias
y
(m/conv_hid0_3x3/bias/Read/ReadVariableOpReadVariableOpm/conv_hid0_3x3/bias*
_output_shapes
:2*
dtype0
�
v/conv_hid0_3x3/kernelVarHandleOp*
_output_shapes
: *'

debug_namev/conv_hid0_3x3/kernel/*
dtype0*
shape:D2*'
shared_namev/conv_hid0_3x3/kernel
�
*v/conv_hid0_3x3/kernel/Read/ReadVariableOpReadVariableOpv/conv_hid0_3x3/kernel*&
_output_shapes
:D2*
dtype0
�
m/conv_hid0_3x3/kernelVarHandleOp*
_output_shapes
: *'

debug_namem/conv_hid0_3x3/kernel/*
dtype0*
shape:D2*'
shared_namem/conv_hid0_3x3/kernel
�
*m/conv_hid0_3x3/kernel/Read/ReadVariableOpReadVariableOpm/conv_hid0_3x3/kernel*&
_output_shapes
:D2*
dtype0
�
v/conv_prep1_5x5/biasVarHandleOp*
_output_shapes
: *&

debug_namev/conv_prep1_5x5/bias/*
dtype0*
shape:*&
shared_namev/conv_prep1_5x5/bias
{
)v/conv_prep1_5x5/bias/Read/ReadVariableOpReadVariableOpv/conv_prep1_5x5/bias*
_output_shapes
:*
dtype0
�
m/conv_prep1_5x5/biasVarHandleOp*
_output_shapes
: *&

debug_namem/conv_prep1_5x5/bias/*
dtype0*
shape:*&
shared_namem/conv_prep1_5x5/bias
{
)m/conv_prep1_5x5/bias/Read/ReadVariableOpReadVariableOpm/conv_prep1_5x5/bias*
_output_shapes
:*
dtype0
�
v/conv_prep1_5x5/kernelVarHandleOp*
_output_shapes
: *(

debug_namev/conv_prep1_5x5/kernel/*
dtype0*
shape:A*(
shared_namev/conv_prep1_5x5/kernel
�
+v/conv_prep1_5x5/kernel/Read/ReadVariableOpReadVariableOpv/conv_prep1_5x5/kernel*&
_output_shapes
:A*
dtype0
�
m/conv_prep1_5x5/kernelVarHandleOp*
_output_shapes
: *(

debug_namem/conv_prep1_5x5/kernel/*
dtype0*
shape:A*(
shared_namem/conv_prep1_5x5/kernel
�
+m/conv_prep1_5x5/kernel/Read/ReadVariableOpReadVariableOpm/conv_prep1_5x5/kernel*&
_output_shapes
:A*
dtype0
�
v/conv_prep1_4x4/biasVarHandleOp*
_output_shapes
: *&

debug_namev/conv_prep1_4x4/bias/*
dtype0*
shape:
*&
shared_namev/conv_prep1_4x4/bias
{
)v/conv_prep1_4x4/bias/Read/ReadVariableOpReadVariableOpv/conv_prep1_4x4/bias*
_output_shapes
:
*
dtype0
�
m/conv_prep1_4x4/biasVarHandleOp*
_output_shapes
: *&

debug_namem/conv_prep1_4x4/bias/*
dtype0*
shape:
*&
shared_namem/conv_prep1_4x4/bias
{
)m/conv_prep1_4x4/bias/Read/ReadVariableOpReadVariableOpm/conv_prep1_4x4/bias*
_output_shapes
:
*
dtype0
�
v/conv_prep1_4x4/kernelVarHandleOp*
_output_shapes
: *(

debug_namev/conv_prep1_4x4/kernel/*
dtype0*
shape:A
*(
shared_namev/conv_prep1_4x4/kernel
�
+v/conv_prep1_4x4/kernel/Read/ReadVariableOpReadVariableOpv/conv_prep1_4x4/kernel*&
_output_shapes
:A
*
dtype0
�
m/conv_prep1_4x4/kernelVarHandleOp*
_output_shapes
: *(

debug_namem/conv_prep1_4x4/kernel/*
dtype0*
shape:A
*(
shared_namem/conv_prep1_4x4/kernel
�
+m/conv_prep1_4x4/kernel/Read/ReadVariableOpReadVariableOpm/conv_prep1_4x4/kernel*&
_output_shapes
:A
*
dtype0
�
v/conv_prep1_3x3/biasVarHandleOp*
_output_shapes
: *&

debug_namev/conv_prep1_3x3/bias/*
dtype0*
shape:2*&
shared_namev/conv_prep1_3x3/bias
{
)v/conv_prep1_3x3/bias/Read/ReadVariableOpReadVariableOpv/conv_prep1_3x3/bias*
_output_shapes
:2*
dtype0
�
m/conv_prep1_3x3/biasVarHandleOp*
_output_shapes
: *&

debug_namem/conv_prep1_3x3/bias/*
dtype0*
shape:2*&
shared_namem/conv_prep1_3x3/bias
{
)m/conv_prep1_3x3/bias/Read/ReadVariableOpReadVariableOpm/conv_prep1_3x3/bias*
_output_shapes
:2*
dtype0
�
v/conv_prep1_3x3/kernelVarHandleOp*
_output_shapes
: *(

debug_namev/conv_prep1_3x3/kernel/*
dtype0*
shape:A2*(
shared_namev/conv_prep1_3x3/kernel
�
+v/conv_prep1_3x3/kernel/Read/ReadVariableOpReadVariableOpv/conv_prep1_3x3/kernel*&
_output_shapes
:A2*
dtype0
�
m/conv_prep1_3x3/kernelVarHandleOp*
_output_shapes
: *(

debug_namem/conv_prep1_3x3/kernel/*
dtype0*
shape:A2*(
shared_namem/conv_prep1_3x3/kernel
�
+m/conv_prep1_3x3/kernel/Read/ReadVariableOpReadVariableOpm/conv_prep1_3x3/kernel*&
_output_shapes
:A2*
dtype0
�
v/conv_prep0_5x5/biasVarHandleOp*
_output_shapes
: *&

debug_namev/conv_prep0_5x5/bias/*
dtype0*
shape:*&
shared_namev/conv_prep0_5x5/bias
{
)v/conv_prep0_5x5/bias/Read/ReadVariableOpReadVariableOpv/conv_prep0_5x5/bias*
_output_shapes
:*
dtype0
�
m/conv_prep0_5x5/biasVarHandleOp*
_output_shapes
: *&

debug_namem/conv_prep0_5x5/bias/*
dtype0*
shape:*&
shared_namem/conv_prep0_5x5/bias
{
)m/conv_prep0_5x5/bias/Read/ReadVariableOpReadVariableOpm/conv_prep0_5x5/bias*
_output_shapes
:*
dtype0
�
v/conv_prep0_5x5/kernelVarHandleOp*
_output_shapes
: *(

debug_namev/conv_prep0_5x5/kernel/*
dtype0*
shape:*(
shared_namev/conv_prep0_5x5/kernel
�
+v/conv_prep0_5x5/kernel/Read/ReadVariableOpReadVariableOpv/conv_prep0_5x5/kernel*&
_output_shapes
:*
dtype0
�
m/conv_prep0_5x5/kernelVarHandleOp*
_output_shapes
: *(

debug_namem/conv_prep0_5x5/kernel/*
dtype0*
shape:*(
shared_namem/conv_prep0_5x5/kernel
�
+m/conv_prep0_5x5/kernel/Read/ReadVariableOpReadVariableOpm/conv_prep0_5x5/kernel*&
_output_shapes
:*
dtype0
�
v/conv_prep0_4x4/biasVarHandleOp*
_output_shapes
: *&

debug_namev/conv_prep0_4x4/bias/*
dtype0*
shape:
*&
shared_namev/conv_prep0_4x4/bias
{
)v/conv_prep0_4x4/bias/Read/ReadVariableOpReadVariableOpv/conv_prep0_4x4/bias*
_output_shapes
:
*
dtype0
�
m/conv_prep0_4x4/biasVarHandleOp*
_output_shapes
: *&

debug_namem/conv_prep0_4x4/bias/*
dtype0*
shape:
*&
shared_namem/conv_prep0_4x4/bias
{
)m/conv_prep0_4x4/bias/Read/ReadVariableOpReadVariableOpm/conv_prep0_4x4/bias*
_output_shapes
:
*
dtype0
�
v/conv_prep0_4x4/kernelVarHandleOp*
_output_shapes
: *(

debug_namev/conv_prep0_4x4/kernel/*
dtype0*
shape:
*(
shared_namev/conv_prep0_4x4/kernel
�
+v/conv_prep0_4x4/kernel/Read/ReadVariableOpReadVariableOpv/conv_prep0_4x4/kernel*&
_output_shapes
:
*
dtype0
�
m/conv_prep0_4x4/kernelVarHandleOp*
_output_shapes
: *(

debug_namem/conv_prep0_4x4/kernel/*
dtype0*
shape:
*(
shared_namem/conv_prep0_4x4/kernel
�
+m/conv_prep0_4x4/kernel/Read/ReadVariableOpReadVariableOpm/conv_prep0_4x4/kernel*&
_output_shapes
:
*
dtype0
�
v/conv_prep0_3x3/biasVarHandleOp*
_output_shapes
: *&

debug_namev/conv_prep0_3x3/bias/*
dtype0*
shape:2*&
shared_namev/conv_prep0_3x3/bias
{
)v/conv_prep0_3x3/bias/Read/ReadVariableOpReadVariableOpv/conv_prep0_3x3/bias*
_output_shapes
:2*
dtype0
�
m/conv_prep0_3x3/biasVarHandleOp*
_output_shapes
: *&

debug_namem/conv_prep0_3x3/bias/*
dtype0*
shape:2*&
shared_namem/conv_prep0_3x3/bias
{
)m/conv_prep0_3x3/bias/Read/ReadVariableOpReadVariableOpm/conv_prep0_3x3/bias*
_output_shapes
:2*
dtype0
�
v/conv_prep0_3x3/kernelVarHandleOp*
_output_shapes
: *(

debug_namev/conv_prep0_3x3/kernel/*
dtype0*
shape:2*(
shared_namev/conv_prep0_3x3/kernel
�
+v/conv_prep0_3x3/kernel/Read/ReadVariableOpReadVariableOpv/conv_prep0_3x3/kernel*&
_output_shapes
:2*
dtype0
�
m/conv_prep0_3x3/kernelVarHandleOp*
_output_shapes
: *(

debug_namem/conv_prep0_3x3/kernel/*
dtype0*
shape:2*(
shared_namem/conv_prep0_3x3/kernel
�
+m/conv_prep0_3x3/kernel/Read/ReadVariableOpReadVariableOpm/conv_prep0_3x3/kernel*&
_output_shapes
:2*
dtype0
�
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
�
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
output_S/biasVarHandleOp*
_output_shapes
: *

debug_nameoutput_S/bias/*
dtype0*
shape:*
shared_nameoutput_S/bias
k
!output_S/bias/Read/ReadVariableOpReadVariableOpoutput_S/bias*
_output_shapes
:*
dtype0
�
output_S/kernelVarHandleOp*
_output_shapes
: * 

debug_nameoutput_S/kernel/*
dtype0*
shape:A* 
shared_nameoutput_S/kernel
{
#output_S/kernel/Read/ReadVariableOpReadVariableOpoutput_S/kernel*&
_output_shapes
:A*
dtype0
�
conv_rev5_5x5/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_rev5_5x5/bias/*
dtype0*
shape:*#
shared_nameconv_rev5_5x5/bias
u
&conv_rev5_5x5/bias/Read/ReadVariableOpReadVariableOpconv_rev5_5x5/bias*
_output_shapes
:*
dtype0
�
conv_rev5_5x5/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_rev5_5x5/kernel/*
dtype0*
shape:A*%
shared_nameconv_rev5_5x5/kernel
�
(conv_rev5_5x5/kernel/Read/ReadVariableOpReadVariableOpconv_rev5_5x5/kernel*&
_output_shapes
:A*
dtype0
�
conv_rev4_4x4/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_rev4_4x4/bias/*
dtype0*
shape:
*#
shared_nameconv_rev4_4x4/bias
u
&conv_rev4_4x4/bias/Read/ReadVariableOpReadVariableOpconv_rev4_4x4/bias*
_output_shapes
:
*
dtype0
�
conv_rev4_4x4/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_rev4_4x4/kernel/*
dtype0*
shape:A
*%
shared_nameconv_rev4_4x4/kernel
�
(conv_rev4_4x4/kernel/Read/ReadVariableOpReadVariableOpconv_rev4_4x4/kernel*&
_output_shapes
:A
*
dtype0
�
conv_rev4_3x3/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_rev4_3x3/bias/*
dtype0*
shape:2*#
shared_nameconv_rev4_3x3/bias
u
&conv_rev4_3x3/bias/Read/ReadVariableOpReadVariableOpconv_rev4_3x3/bias*
_output_shapes
:2*
dtype0
�
conv_rev4_3x3/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_rev4_3x3/kernel/*
dtype0*
shape:A2*%
shared_nameconv_rev4_3x3/kernel
�
(conv_rev4_3x3/kernel/Read/ReadVariableOpReadVariableOpconv_rev4_3x3/kernel*&
_output_shapes
:A2*
dtype0
�
conv_rev3_5x5/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_rev3_5x5/bias/*
dtype0*
shape:*#
shared_nameconv_rev3_5x5/bias
u
&conv_rev3_5x5/bias/Read/ReadVariableOpReadVariableOpconv_rev3_5x5/bias*
_output_shapes
:*
dtype0
�
conv_rev3_5x5/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_rev3_5x5/kernel/*
dtype0*
shape:A*%
shared_nameconv_rev3_5x5/kernel
�
(conv_rev3_5x5/kernel/Read/ReadVariableOpReadVariableOpconv_rev3_5x5/kernel*&
_output_shapes
:A*
dtype0
�
conv_rev3_4x4/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_rev3_4x4/bias/*
dtype0*
shape:
*#
shared_nameconv_rev3_4x4/bias
u
&conv_rev3_4x4/bias/Read/ReadVariableOpReadVariableOpconv_rev3_4x4/bias*
_output_shapes
:
*
dtype0
�
conv_rev3_4x4/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_rev3_4x4/kernel/*
dtype0*
shape:A
*%
shared_nameconv_rev3_4x4/kernel
�
(conv_rev3_4x4/kernel/Read/ReadVariableOpReadVariableOpconv_rev3_4x4/kernel*&
_output_shapes
:A
*
dtype0
�
conv_rev3_3x3/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_rev3_3x3/bias/*
dtype0*
shape:2*#
shared_nameconv_rev3_3x3/bias
u
&conv_rev3_3x3/bias/Read/ReadVariableOpReadVariableOpconv_rev3_3x3/bias*
_output_shapes
:2*
dtype0
�
conv_rev3_3x3/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_rev3_3x3/kernel/*
dtype0*
shape:A2*%
shared_nameconv_rev3_3x3/kernel
�
(conv_rev3_3x3/kernel/Read/ReadVariableOpReadVariableOpconv_rev3_3x3/kernel*&
_output_shapes
:A2*
dtype0
�
conv_rev2_5x5/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_rev2_5x5/bias/*
dtype0*
shape:*#
shared_nameconv_rev2_5x5/bias
u
&conv_rev2_5x5/bias/Read/ReadVariableOpReadVariableOpconv_rev2_5x5/bias*
_output_shapes
:*
dtype0
�
conv_rev2_5x5/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_rev2_5x5/kernel/*
dtype0*
shape:A*%
shared_nameconv_rev2_5x5/kernel
�
(conv_rev2_5x5/kernel/Read/ReadVariableOpReadVariableOpconv_rev2_5x5/kernel*&
_output_shapes
:A*
dtype0
�
conv_rev2_4x4/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_rev2_4x4/bias/*
dtype0*
shape:
*#
shared_nameconv_rev2_4x4/bias
u
&conv_rev2_4x4/bias/Read/ReadVariableOpReadVariableOpconv_rev2_4x4/bias*
_output_shapes
:
*
dtype0
�
conv_rev2_4x4/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_rev2_4x4/kernel/*
dtype0*
shape:A
*%
shared_nameconv_rev2_4x4/kernel
�
(conv_rev2_4x4/kernel/Read/ReadVariableOpReadVariableOpconv_rev2_4x4/kernel*&
_output_shapes
:A
*
dtype0
�
conv_rev2_3x3/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_rev2_3x3/bias/*
dtype0*
shape:2*#
shared_nameconv_rev2_3x3/bias
u
&conv_rev2_3x3/bias/Read/ReadVariableOpReadVariableOpconv_rev2_3x3/bias*
_output_shapes
:2*
dtype0
�
conv_rev2_3x3/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_rev2_3x3/kernel/*
dtype0*
shape:A2*%
shared_nameconv_rev2_3x3/kernel
�
(conv_rev2_3x3/kernel/Read/ReadVariableOpReadVariableOpconv_rev2_3x3/kernel*&
_output_shapes
:A2*
dtype0
�
conv_rev1_5x5/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_rev1_5x5/bias/*
dtype0*
shape:*#
shared_nameconv_rev1_5x5/bias
u
&conv_rev1_5x5/bias/Read/ReadVariableOpReadVariableOpconv_rev1_5x5/bias*
_output_shapes
:*
dtype0
�
conv_rev1_5x5/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_rev1_5x5/kernel/*
dtype0*
shape:A*%
shared_nameconv_rev1_5x5/kernel
�
(conv_rev1_5x5/kernel/Read/ReadVariableOpReadVariableOpconv_rev1_5x5/kernel*&
_output_shapes
:A*
dtype0
�
conv_rev1_4x4/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_rev1_4x4/bias/*
dtype0*
shape:
*#
shared_nameconv_rev1_4x4/bias
u
&conv_rev1_4x4/bias/Read/ReadVariableOpReadVariableOpconv_rev1_4x4/bias*
_output_shapes
:
*
dtype0
�
conv_rev1_4x4/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_rev1_4x4/kernel/*
dtype0*
shape:A
*%
shared_nameconv_rev1_4x4/kernel
�
(conv_rev1_4x4/kernel/Read/ReadVariableOpReadVariableOpconv_rev1_4x4/kernel*&
_output_shapes
:A
*
dtype0
�
conv_rev1_3x3/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_rev1_3x3/bias/*
dtype0*
shape:2*#
shared_nameconv_rev1_3x3/bias
u
&conv_rev1_3x3/bias/Read/ReadVariableOpReadVariableOpconv_rev1_3x3/bias*
_output_shapes
:2*
dtype0
�
conv_rev1_3x3/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_rev1_3x3/kernel/*
dtype0*
shape:A2*%
shared_nameconv_rev1_3x3/kernel
�
(conv_rev1_3x3/kernel/Read/ReadVariableOpReadVariableOpconv_rev1_3x3/kernel*&
_output_shapes
:A2*
dtype0
�
conv_rev0_5x5/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_rev0_5x5/bias/*
dtype0*
shape:*#
shared_nameconv_rev0_5x5/bias
u
&conv_rev0_5x5/bias/Read/ReadVariableOpReadVariableOpconv_rev0_5x5/bias*
_output_shapes
:*
dtype0
�
conv_rev0_5x5/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_rev0_5x5/kernel/*
dtype0*
shape:*%
shared_nameconv_rev0_5x5/kernel
�
(conv_rev0_5x5/kernel/Read/ReadVariableOpReadVariableOpconv_rev0_5x5/kernel*&
_output_shapes
:*
dtype0
�
conv_rev0_4x4/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_rev0_4x4/bias/*
dtype0*
shape:
*#
shared_nameconv_rev0_4x4/bias
u
&conv_rev0_4x4/bias/Read/ReadVariableOpReadVariableOpconv_rev0_4x4/bias*
_output_shapes
:
*
dtype0
�
conv_rev0_4x4/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_rev0_4x4/kernel/*
dtype0*
shape:
*%
shared_nameconv_rev0_4x4/kernel
�
(conv_rev0_4x4/kernel/Read/ReadVariableOpReadVariableOpconv_rev0_4x4/kernel*&
_output_shapes
:
*
dtype0
�
conv_rev0_3x3/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_rev0_3x3/bias/*
dtype0*
shape:2*#
shared_nameconv_rev0_3x3/bias
u
&conv_rev0_3x3/bias/Read/ReadVariableOpReadVariableOpconv_rev0_3x3/bias*
_output_shapes
:2*
dtype0
�
conv_rev0_3x3/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_rev0_3x3/kernel/*
dtype0*
shape:2*%
shared_nameconv_rev0_3x3/kernel
�
(conv_rev0_3x3/kernel/Read/ReadVariableOpReadVariableOpconv_rev0_3x3/kernel*&
_output_shapes
:2*
dtype0
�
output_C/biasVarHandleOp*
_output_shapes
: *

debug_nameoutput_C/bias/*
dtype0*
shape:*
shared_nameoutput_C/bias
k
!output_C/bias/Read/ReadVariableOpReadVariableOpoutput_C/bias*
_output_shapes
:*
dtype0
�
output_C/kernelVarHandleOp*
_output_shapes
: * 

debug_nameoutput_C/kernel/*
dtype0*
shape:A* 
shared_nameoutput_C/kernel
{
#output_C/kernel/Read/ReadVariableOpReadVariableOpoutput_C/kernel*&
_output_shapes
:A*
dtype0
�
conv_hid5_5x5/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_hid5_5x5/bias/*
dtype0*
shape:*#
shared_nameconv_hid5_5x5/bias
u
&conv_hid5_5x5/bias/Read/ReadVariableOpReadVariableOpconv_hid5_5x5/bias*
_output_shapes
:*
dtype0
�
conv_hid5_5x5/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_hid5_5x5/kernel/*
dtype0*
shape:A*%
shared_nameconv_hid5_5x5/kernel
�
(conv_hid5_5x5/kernel/Read/ReadVariableOpReadVariableOpconv_hid5_5x5/kernel*&
_output_shapes
:A*
dtype0
�
conv_hid4_4x4/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_hid4_4x4/bias/*
dtype0*
shape:
*#
shared_nameconv_hid4_4x4/bias
u
&conv_hid4_4x4/bias/Read/ReadVariableOpReadVariableOpconv_hid4_4x4/bias*
_output_shapes
:
*
dtype0
�
conv_hid4_4x4/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_hid4_4x4/kernel/*
dtype0*
shape:A
*%
shared_nameconv_hid4_4x4/kernel
�
(conv_hid4_4x4/kernel/Read/ReadVariableOpReadVariableOpconv_hid4_4x4/kernel*&
_output_shapes
:A
*
dtype0
�
conv_hid4_3x3/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_hid4_3x3/bias/*
dtype0*
shape:2*#
shared_nameconv_hid4_3x3/bias
u
&conv_hid4_3x3/bias/Read/ReadVariableOpReadVariableOpconv_hid4_3x3/bias*
_output_shapes
:2*
dtype0
�
conv_hid4_3x3/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_hid4_3x3/kernel/*
dtype0*
shape:A2*%
shared_nameconv_hid4_3x3/kernel
�
(conv_hid4_3x3/kernel/Read/ReadVariableOpReadVariableOpconv_hid4_3x3/kernel*&
_output_shapes
:A2*
dtype0
�
conv_hid3_5x5/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_hid3_5x5/bias/*
dtype0*
shape:*#
shared_nameconv_hid3_5x5/bias
u
&conv_hid3_5x5/bias/Read/ReadVariableOpReadVariableOpconv_hid3_5x5/bias*
_output_shapes
:*
dtype0
�
conv_hid3_5x5/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_hid3_5x5/kernel/*
dtype0*
shape:A*%
shared_nameconv_hid3_5x5/kernel
�
(conv_hid3_5x5/kernel/Read/ReadVariableOpReadVariableOpconv_hid3_5x5/kernel*&
_output_shapes
:A*
dtype0
�
conv_hid3_4x4/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_hid3_4x4/bias/*
dtype0*
shape:
*#
shared_nameconv_hid3_4x4/bias
u
&conv_hid3_4x4/bias/Read/ReadVariableOpReadVariableOpconv_hid3_4x4/bias*
_output_shapes
:
*
dtype0
�
conv_hid3_4x4/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_hid3_4x4/kernel/*
dtype0*
shape:A
*%
shared_nameconv_hid3_4x4/kernel
�
(conv_hid3_4x4/kernel/Read/ReadVariableOpReadVariableOpconv_hid3_4x4/kernel*&
_output_shapes
:A
*
dtype0
�
conv_hid3_3x3/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_hid3_3x3/bias/*
dtype0*
shape:2*#
shared_nameconv_hid3_3x3/bias
u
&conv_hid3_3x3/bias/Read/ReadVariableOpReadVariableOpconv_hid3_3x3/bias*
_output_shapes
:2*
dtype0
�
conv_hid3_3x3/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_hid3_3x3/kernel/*
dtype0*
shape:A2*%
shared_nameconv_hid3_3x3/kernel
�
(conv_hid3_3x3/kernel/Read/ReadVariableOpReadVariableOpconv_hid3_3x3/kernel*&
_output_shapes
:A2*
dtype0
�
conv_hid2_5x5/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_hid2_5x5/bias/*
dtype0*
shape:*#
shared_nameconv_hid2_5x5/bias
u
&conv_hid2_5x5/bias/Read/ReadVariableOpReadVariableOpconv_hid2_5x5/bias*
_output_shapes
:*
dtype0
�
conv_hid2_5x5/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_hid2_5x5/kernel/*
dtype0*
shape:A*%
shared_nameconv_hid2_5x5/kernel
�
(conv_hid2_5x5/kernel/Read/ReadVariableOpReadVariableOpconv_hid2_5x5/kernel*&
_output_shapes
:A*
dtype0
�
conv_hid2_4x4/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_hid2_4x4/bias/*
dtype0*
shape:
*#
shared_nameconv_hid2_4x4/bias
u
&conv_hid2_4x4/bias/Read/ReadVariableOpReadVariableOpconv_hid2_4x4/bias*
_output_shapes
:
*
dtype0
�
conv_hid2_4x4/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_hid2_4x4/kernel/*
dtype0*
shape:A
*%
shared_nameconv_hid2_4x4/kernel
�
(conv_hid2_4x4/kernel/Read/ReadVariableOpReadVariableOpconv_hid2_4x4/kernel*&
_output_shapes
:A
*
dtype0
�
conv_hid2_3x3/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_hid2_3x3/bias/*
dtype0*
shape:2*#
shared_nameconv_hid2_3x3/bias
u
&conv_hid2_3x3/bias/Read/ReadVariableOpReadVariableOpconv_hid2_3x3/bias*
_output_shapes
:2*
dtype0
�
conv_hid2_3x3/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_hid2_3x3/kernel/*
dtype0*
shape:A2*%
shared_nameconv_hid2_3x3/kernel
�
(conv_hid2_3x3/kernel/Read/ReadVariableOpReadVariableOpconv_hid2_3x3/kernel*&
_output_shapes
:A2*
dtype0
�
conv_hid1_5x5/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_hid1_5x5/bias/*
dtype0*
shape:*#
shared_nameconv_hid1_5x5/bias
u
&conv_hid1_5x5/bias/Read/ReadVariableOpReadVariableOpconv_hid1_5x5/bias*
_output_shapes
:*
dtype0
�
conv_hid1_5x5/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_hid1_5x5/kernel/*
dtype0*
shape:A*%
shared_nameconv_hid1_5x5/kernel
�
(conv_hid1_5x5/kernel/Read/ReadVariableOpReadVariableOpconv_hid1_5x5/kernel*&
_output_shapes
:A*
dtype0
�
conv_hid1_4x4/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_hid1_4x4/bias/*
dtype0*
shape:
*#
shared_nameconv_hid1_4x4/bias
u
&conv_hid1_4x4/bias/Read/ReadVariableOpReadVariableOpconv_hid1_4x4/bias*
_output_shapes
:
*
dtype0
�
conv_hid1_4x4/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_hid1_4x4/kernel/*
dtype0*
shape:A
*%
shared_nameconv_hid1_4x4/kernel
�
(conv_hid1_4x4/kernel/Read/ReadVariableOpReadVariableOpconv_hid1_4x4/kernel*&
_output_shapes
:A
*
dtype0
�
conv_hid1_3x3/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_hid1_3x3/bias/*
dtype0*
shape:2*#
shared_nameconv_hid1_3x3/bias
u
&conv_hid1_3x3/bias/Read/ReadVariableOpReadVariableOpconv_hid1_3x3/bias*
_output_shapes
:2*
dtype0
�
conv_hid1_3x3/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_hid1_3x3/kernel/*
dtype0*
shape:A2*%
shared_nameconv_hid1_3x3/kernel
�
(conv_hid1_3x3/kernel/Read/ReadVariableOpReadVariableOpconv_hid1_3x3/kernel*&
_output_shapes
:A2*
dtype0
�
conv_hid0_5x5/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_hid0_5x5/bias/*
dtype0*
shape:*#
shared_nameconv_hid0_5x5/bias
u
&conv_hid0_5x5/bias/Read/ReadVariableOpReadVariableOpconv_hid0_5x5/bias*
_output_shapes
:*
dtype0
�
conv_hid0_5x5/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_hid0_5x5/kernel/*
dtype0*
shape:D*%
shared_nameconv_hid0_5x5/kernel
�
(conv_hid0_5x5/kernel/Read/ReadVariableOpReadVariableOpconv_hid0_5x5/kernel*&
_output_shapes
:D*
dtype0
�
conv_hid0_4x4/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_hid0_4x4/bias/*
dtype0*
shape:
*#
shared_nameconv_hid0_4x4/bias
u
&conv_hid0_4x4/bias/Read/ReadVariableOpReadVariableOpconv_hid0_4x4/bias*
_output_shapes
:
*
dtype0
�
conv_hid0_4x4/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_hid0_4x4/kernel/*
dtype0*
shape:D
*%
shared_nameconv_hid0_4x4/kernel
�
(conv_hid0_4x4/kernel/Read/ReadVariableOpReadVariableOpconv_hid0_4x4/kernel*&
_output_shapes
:D
*
dtype0
�
conv_hid0_3x3/biasVarHandleOp*
_output_shapes
: *#

debug_nameconv_hid0_3x3/bias/*
dtype0*
shape:2*#
shared_nameconv_hid0_3x3/bias
u
&conv_hid0_3x3/bias/Read/ReadVariableOpReadVariableOpconv_hid0_3x3/bias*
_output_shapes
:2*
dtype0
�
conv_hid0_3x3/kernelVarHandleOp*
_output_shapes
: *%

debug_nameconv_hid0_3x3/kernel/*
dtype0*
shape:D2*%
shared_nameconv_hid0_3x3/kernel
�
(conv_hid0_3x3/kernel/Read/ReadVariableOpReadVariableOpconv_hid0_3x3/kernel*&
_output_shapes
:D2*
dtype0
�
conv_prep1_5x5/biasVarHandleOp*
_output_shapes
: *$

debug_nameconv_prep1_5x5/bias/*
dtype0*
shape:*$
shared_nameconv_prep1_5x5/bias
w
'conv_prep1_5x5/bias/Read/ReadVariableOpReadVariableOpconv_prep1_5x5/bias*
_output_shapes
:*
dtype0
�
conv_prep1_5x5/kernelVarHandleOp*
_output_shapes
: *&

debug_nameconv_prep1_5x5/kernel/*
dtype0*
shape:A*&
shared_nameconv_prep1_5x5/kernel
�
)conv_prep1_5x5/kernel/Read/ReadVariableOpReadVariableOpconv_prep1_5x5/kernel*&
_output_shapes
:A*
dtype0
�
conv_prep1_4x4/biasVarHandleOp*
_output_shapes
: *$

debug_nameconv_prep1_4x4/bias/*
dtype0*
shape:
*$
shared_nameconv_prep1_4x4/bias
w
'conv_prep1_4x4/bias/Read/ReadVariableOpReadVariableOpconv_prep1_4x4/bias*
_output_shapes
:
*
dtype0
�
conv_prep1_4x4/kernelVarHandleOp*
_output_shapes
: *&

debug_nameconv_prep1_4x4/kernel/*
dtype0*
shape:A
*&
shared_nameconv_prep1_4x4/kernel
�
)conv_prep1_4x4/kernel/Read/ReadVariableOpReadVariableOpconv_prep1_4x4/kernel*&
_output_shapes
:A
*
dtype0
�
conv_prep1_3x3/biasVarHandleOp*
_output_shapes
: *$

debug_nameconv_prep1_3x3/bias/*
dtype0*
shape:2*$
shared_nameconv_prep1_3x3/bias
w
'conv_prep1_3x3/bias/Read/ReadVariableOpReadVariableOpconv_prep1_3x3/bias*
_output_shapes
:2*
dtype0
�
conv_prep1_3x3/kernelVarHandleOp*
_output_shapes
: *&

debug_nameconv_prep1_3x3/kernel/*
dtype0*
shape:A2*&
shared_nameconv_prep1_3x3/kernel
�
)conv_prep1_3x3/kernel/Read/ReadVariableOpReadVariableOpconv_prep1_3x3/kernel*&
_output_shapes
:A2*
dtype0
�
conv_prep0_5x5/biasVarHandleOp*
_output_shapes
: *$

debug_nameconv_prep0_5x5/bias/*
dtype0*
shape:*$
shared_nameconv_prep0_5x5/bias
w
'conv_prep0_5x5/bias/Read/ReadVariableOpReadVariableOpconv_prep0_5x5/bias*
_output_shapes
:*
dtype0
�
conv_prep0_5x5/kernelVarHandleOp*
_output_shapes
: *&

debug_nameconv_prep0_5x5/kernel/*
dtype0*
shape:*&
shared_nameconv_prep0_5x5/kernel
�
)conv_prep0_5x5/kernel/Read/ReadVariableOpReadVariableOpconv_prep0_5x5/kernel*&
_output_shapes
:*
dtype0
�
conv_prep0_4x4/biasVarHandleOp*
_output_shapes
: *$

debug_nameconv_prep0_4x4/bias/*
dtype0*
shape:
*$
shared_nameconv_prep0_4x4/bias
w
'conv_prep0_4x4/bias/Read/ReadVariableOpReadVariableOpconv_prep0_4x4/bias*
_output_shapes
:
*
dtype0
�
conv_prep0_4x4/kernelVarHandleOp*
_output_shapes
: *&

debug_nameconv_prep0_4x4/kernel/*
dtype0*
shape:
*&
shared_nameconv_prep0_4x4/kernel
�
)conv_prep0_4x4/kernel/Read/ReadVariableOpReadVariableOpconv_prep0_4x4/kernel*&
_output_shapes
:
*
dtype0
�
conv_prep0_3x3/biasVarHandleOp*
_output_shapes
: *$

debug_nameconv_prep0_3x3/bias/*
dtype0*
shape:2*$
shared_nameconv_prep0_3x3/bias
w
'conv_prep0_3x3/bias/Read/ReadVariableOpReadVariableOpconv_prep0_3x3/bias*
_output_shapes
:2*
dtype0
�
conv_prep0_3x3/kernelVarHandleOp*
_output_shapes
: *&

debug_nameconv_prep0_3x3/kernel/*
dtype0*
shape:2*&
shared_nameconv_prep0_3x3/kernel
�
)conv_prep0_3x3/kernel/Read/ReadVariableOpReadVariableOpconv_prep0_3x3/kernel*&
_output_shapes
:2*
dtype0
�
serving_default_input_1Placeholder*/
_output_shapes
:���������@@*
dtype0*$
shape:���������@@
�
serving_default_input_2Placeholder*/
_output_shapes
:���������@@*
dtype0*$
shape:���������@@
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2conv_prep0_3x3/kernelconv_prep0_3x3/biasconv_prep0_4x4/kernelconv_prep0_4x4/biasconv_prep0_5x5/kernelconv_prep0_5x5/biasconv_prep1_3x3/kernelconv_prep1_3x3/biasconv_prep1_4x4/kernelconv_prep1_4x4/biasconv_prep1_5x5/kernelconv_prep1_5x5/biasconv_hid0_3x3/kernelconv_hid0_3x3/biasconv_hid0_4x4/kernelconv_hid0_4x4/biasconv_hid0_5x5/kernelconv_hid0_5x5/biasconv_hid1_3x3/kernelconv_hid1_3x3/biasconv_hid1_4x4/kernelconv_hid1_4x4/biasconv_hid1_5x5/kernelconv_hid1_5x5/biasconv_hid2_3x3/kernelconv_hid2_3x3/biasconv_hid2_4x4/kernelconv_hid2_4x4/biasconv_hid2_5x5/kernelconv_hid2_5x5/biasconv_hid3_3x3/kernelconv_hid3_3x3/biasconv_hid3_4x4/kernelconv_hid3_4x4/biasconv_hid3_5x5/kernelconv_hid3_5x5/biasconv_hid4_3x3/kernelconv_hid4_3x3/biasconv_hid4_4x4/kernelconv_hid4_4x4/biasconv_hid5_5x5/kernelconv_hid5_5x5/biasoutput_C/kerneloutput_C/biasconv_rev0_3x3/kernelconv_rev0_3x3/biasconv_rev0_4x4/kernelconv_rev0_4x4/biasconv_rev0_5x5/kernelconv_rev0_5x5/biasconv_rev1_3x3/kernelconv_rev1_3x3/biasconv_rev1_4x4/kernelconv_rev1_4x4/biasconv_rev1_5x5/kernelconv_rev1_5x5/biasconv_rev2_3x3/kernelconv_rev2_3x3/biasconv_rev2_4x4/kernelconv_rev2_4x4/biasconv_rev2_5x5/kernelconv_rev2_5x5/biasconv_rev3_3x3/kernelconv_rev3_3x3/biasconv_rev3_4x4/kernelconv_rev3_4x4/biasconv_rev3_5x5/kernelconv_rev3_5x5/biasconv_rev4_3x3/kernelconv_rev4_3x3/biasconv_rev4_4x4/kernelconv_rev4_4x4/biasconv_rev5_5x5/kernelconv_rev5_5x5/biasoutput_S/kerneloutput_S/bias*Y
TinR
P2N*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*n
_read_only_resource_inputsP
NL	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLM*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_4811

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value�B� B�
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
* 
�	
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
layer-8
layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
 layer_with_weights-11
 layer-17
!layer-18
"layer_with_weights-12
"layer-19
#layer_with_weights-13
#layer-20
$layer_with_weights-14
$layer-21
%layer-22
&layer_with_weights-15
&layer-23
'layer_with_weights-16
'layer-24
(layer_with_weights-17
(layer-25
)layer-26
*layer_with_weights-18
*layer-27
+layer_with_weights-19
+layer-28
,layer_with_weights-20
,layer-29
-layer-30
.layer_with_weights-21
.layer-31
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses*
�
5layer-0
6layer-1
7layer_with_weights-0
7layer-2
8layer_with_weights-1
8layer-3
9layer_with_weights-2
9layer-4
:layer-5
;layer_with_weights-3
;layer-6
<layer_with_weights-4
<layer-7
=layer_with_weights-5
=layer-8
>layer-9
?layer_with_weights-6
?layer-10
@layer_with_weights-7
@layer-11
Alayer_with_weights-8
Alayer-12
Blayer-13
Clayer_with_weights-9
Clayer-14
Dlayer_with_weights-10
Dlayer-15
Elayer_with_weights-11
Elayer-16
Flayer-17
Glayer_with_weights-12
Glayer-18
Hlayer_with_weights-13
Hlayer-19
Ilayer_with_weights-14
Ilayer-20
Jlayer-21
Klayer_with_weights-15
Klayer-22
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses*
�
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses* 
�
X0
Y1
Z2
[3
\4
]5
^6
_7
`8
a9
b10
c11
d12
e13
f14
g15
h16
i17
j18
k19
l20
m21
n22
o23
p24
q25
r26
s27
t28
u29
v30
w31
x32
y33
z34
{35
|36
}37
~38
39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72
�73
�74
�75*
�
X0
Y1
Z2
[3
\4
]5
^6
_7
`8
a9
b10
c11
d12
e13
f14
g15
h16
i17
j18
k19
l20
m21
n22
o23
p24
q25
r26
s27
t28
u29
v30
w31
x32
y33
z34
{35
|36
}37
~38
39
�40
�41
�42
�43*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Xkernel
Ybias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Zkernel
[bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

\kernel
]bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

^kernel
_bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

`kernel
abias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

bkernel
cbias
!�_jit_compiled_convolution_op*
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

dkernel
ebias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

fkernel
gbias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

hkernel
ibias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

jkernel
kbias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

lkernel
mbias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

nkernel
obias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

pkernel
qbias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

rkernel
sbias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

tkernel
ubias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

vkernel
wbias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

xkernel
ybias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

zkernel
{bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

|kernel
}bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

~kernel
bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
X0
Y1
Z2
[3
\4
]5
^6
_7
`8
a9
b10
c11
d12
e13
f14
g15
h16
i17
j18
k19
l20
m21
n22
o23
p24
q25
r26
s27
t28
u29
v30
w31
x32
y33
z34
{35
|36
}37
~38
39
�40
�41
�42
�43*
�
X0
Y1
Z2
[3
\4
]5
^6
_7
`8
a9
b10
c11
d12
e13
f14
g15
h16
i17
j18
k19
l20
m21
n22
o23
p24
q25
r26
s27
t28
u29
v30
w31
x32
y33
z34
{35
|36
}37
~38
39
�40
�41
�42
�43*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
UO
VARIABLE_VALUEconv_prep0_3x3/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_prep0_3x3/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_prep0_4x4/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_prep0_4x4/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_prep0_5x5/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_prep0_5x5/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_prep1_3x3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_prep1_3x3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_prep1_4x4/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_prep1_4x4/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEconv_prep1_5x5/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEconv_prep1_5x5/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_hid0_3x3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_hid0_3x3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_hid0_4x4/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_hid0_4x4/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_hid0_5x5/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_hid0_5x5/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_hid1_3x3/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_hid1_3x3/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_hid1_4x4/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_hid1_4x4/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_hid1_5x5/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_hid1_5x5/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_hid2_3x3/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_hid2_3x3/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_hid2_4x4/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_hid2_4x4/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_hid2_5x5/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_hid2_5x5/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_hid3_3x3/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_hid3_3x3/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_hid3_4x4/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_hid3_4x4/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_hid3_5x5/kernel'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_hid3_5x5/bias'variables/35/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_hid4_3x3/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_hid4_3x3/bias'variables/37/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_hid4_4x4/kernel'variables/38/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_hid4_4x4/bias'variables/39/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_hid5_5x5/kernel'variables/40/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_hid5_5x5/bias'variables/41/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEoutput_C/kernel'variables/42/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEoutput_C/bias'variables/43/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_rev0_3x3/kernel'variables/44/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_rev0_3x3/bias'variables/45/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_rev0_4x4/kernel'variables/46/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_rev0_4x4/bias'variables/47/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_rev0_5x5/kernel'variables/48/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_rev0_5x5/bias'variables/49/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_rev1_3x3/kernel'variables/50/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_rev1_3x3/bias'variables/51/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_rev1_4x4/kernel'variables/52/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_rev1_4x4/bias'variables/53/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_rev1_5x5/kernel'variables/54/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_rev1_5x5/bias'variables/55/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_rev2_3x3/kernel'variables/56/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_rev2_3x3/bias'variables/57/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_rev2_4x4/kernel'variables/58/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_rev2_4x4/bias'variables/59/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_rev2_5x5/kernel'variables/60/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_rev2_5x5/bias'variables/61/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_rev3_3x3/kernel'variables/62/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_rev3_3x3/bias'variables/63/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_rev3_4x4/kernel'variables/64/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_rev3_4x4/bias'variables/65/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_rev3_5x5/kernel'variables/66/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_rev3_5x5/bias'variables/67/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_rev4_3x3/kernel'variables/68/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_rev4_3x3/bias'variables/69/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_rev4_4x4/kernel'variables/70/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_rev4_4x4/bias'variables/71/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_rev5_5x5/kernel'variables/72/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_rev5_5x5/bias'variables/73/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEoutput_S/kernel'variables/74/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEoutput_S/bias'variables/75/.ATTRIBUTES/VARIABLE_VALUE*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31*
'
0
1
2
3
4*

�0*
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72
�73
�74
�75
�76
�77
�78
�79
�80
�81
�82
�83
�84
�85
�86
�87
�88*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43*
* 
* 

X0
Y1*

X0
Y1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

Z0
[1*

Z0
[1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

\0
]1*

\0
]1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

^0
_1*

^0
_1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

`0
a1*

`0
a1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

b0
c1*

b0
c1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

d0
e1*

d0
e1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

f0
g1*

f0
g1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

h0
i1*

h0
i1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

j0
k1*

j0
k1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

l0
m1*

l0
m1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

n0
o1*

n0
o1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

p0
q1*

p0
q1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

r0
s1*

r0
s1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

t0
u1*

t0
u1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

v0
w1*

v0
w1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

x0
y1*

x0
y1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

z0
{1*

z0
{1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

|0
}1*

|0
}1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

~0
1*

~0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
 17
!18
"19
#20
$21
%22
&23
'24
(25
)26
*27
+28
,29
-30
.31*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31*
�
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
D15
E16
F17
G18
H19
I20
J21
K22*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
b\
VARIABLE_VALUEm/conv_prep0_3x3/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEv/conv_prep0_3x3/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEm/conv_prep0_3x3/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEv/conv_prep0_3x3/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEm/conv_prep0_4x4/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEv/conv_prep0_4x4/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEm/conv_prep0_4x4/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEv/conv_prep0_4x4/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEm/conv_prep0_5x5/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEv/conv_prep0_5x5/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEm/conv_prep0_5x5/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEv/conv_prep0_5x5/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEm/conv_prep1_3x3/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEv/conv_prep1_3x3/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEm/conv_prep1_3x3/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEv/conv_prep1_3x3/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEm/conv_prep1_4x4/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEv/conv_prep1_4x4/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEm/conv_prep1_4x4/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEv/conv_prep1_4x4/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEm/conv_prep1_5x5/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEv/conv_prep1_5x5/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEm/conv_prep1_5x5/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEv/conv_prep1_5x5/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEm/conv_hid0_3x3/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEv/conv_hid0_3x3/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEm/conv_hid0_3x3/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEv/conv_hid0_3x3/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEm/conv_hid0_4x4/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEv/conv_hid0_4x4/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEm/conv_hid0_4x4/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEv/conv_hid0_4x4/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEm/conv_hid0_5x5/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEv/conv_hid0_5x5/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEm/conv_hid0_5x5/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEv/conv_hid0_5x5/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEm/conv_hid1_3x3/kernel2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEv/conv_hid1_3x3/kernel2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEm/conv_hid1_3x3/bias2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEv/conv_hid1_3x3/bias2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEm/conv_hid1_4x4/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEv/conv_hid1_4x4/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEm/conv_hid1_4x4/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEv/conv_hid1_4x4/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEm/conv_hid1_5x5/kernel2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEv/conv_hid1_5x5/kernel2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEm/conv_hid1_5x5/bias2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEv/conv_hid1_5x5/bias2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEm/conv_hid2_3x3/kernel2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEv/conv_hid2_3x3/kernel2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEm/conv_hid2_3x3/bias2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEv/conv_hid2_3x3/bias2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEm/conv_hid2_4x4/kernel2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEv/conv_hid2_4x4/kernel2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEm/conv_hid2_4x4/bias2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEv/conv_hid2_4x4/bias2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEm/conv_hid2_5x5/kernel2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEv/conv_hid2_5x5/kernel2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEm/conv_hid2_5x5/bias2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEv/conv_hid2_5x5/bias2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEm/conv_hid3_3x3/kernel2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEv/conv_hid3_3x3/kernel2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEm/conv_hid3_3x3/bias2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEv/conv_hid3_3x3/bias2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEm/conv_hid3_4x4/kernel2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEv/conv_hid3_4x4/kernel2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEm/conv_hid3_4x4/bias2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEv/conv_hid3_4x4/bias2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEm/conv_hid3_5x5/kernel2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEv/conv_hid3_5x5/kernel2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEm/conv_hid3_5x5/bias2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEv/conv_hid3_5x5/bias2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEm/conv_hid4_3x3/kernel2optimizer/_variables/73/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEv/conv_hid4_3x3/kernel2optimizer/_variables/74/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEm/conv_hid4_3x3/bias2optimizer/_variables/75/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEv/conv_hid4_3x3/bias2optimizer/_variables/76/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEm/conv_hid4_4x4/kernel2optimizer/_variables/77/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEv/conv_hid4_4x4/kernel2optimizer/_variables/78/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEm/conv_hid4_4x4/bias2optimizer/_variables/79/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEv/conv_hid4_4x4/bias2optimizer/_variables/80/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEm/conv_hid5_5x5/kernel2optimizer/_variables/81/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEv/conv_hid5_5x5/kernel2optimizer/_variables/82/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEm/conv_hid5_5x5/bias2optimizer/_variables/83/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEv/conv_hid5_5x5/bias2optimizer/_variables/84/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEm/output_C/kernel2optimizer/_variables/85/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEv/output_C/kernel2optimizer/_variables/86/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEm/output_C/bias2optimizer/_variables/87/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEv/output_C/bias2optimizer/_variables/88/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
� 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv_prep0_3x3/kernelconv_prep0_3x3/biasconv_prep0_4x4/kernelconv_prep0_4x4/biasconv_prep0_5x5/kernelconv_prep0_5x5/biasconv_prep1_3x3/kernelconv_prep1_3x3/biasconv_prep1_4x4/kernelconv_prep1_4x4/biasconv_prep1_5x5/kernelconv_prep1_5x5/biasconv_hid0_3x3/kernelconv_hid0_3x3/biasconv_hid0_4x4/kernelconv_hid0_4x4/biasconv_hid0_5x5/kernelconv_hid0_5x5/biasconv_hid1_3x3/kernelconv_hid1_3x3/biasconv_hid1_4x4/kernelconv_hid1_4x4/biasconv_hid1_5x5/kernelconv_hid1_5x5/biasconv_hid2_3x3/kernelconv_hid2_3x3/biasconv_hid2_4x4/kernelconv_hid2_4x4/biasconv_hid2_5x5/kernelconv_hid2_5x5/biasconv_hid3_3x3/kernelconv_hid3_3x3/biasconv_hid3_4x4/kernelconv_hid3_4x4/biasconv_hid3_5x5/kernelconv_hid3_5x5/biasconv_hid4_3x3/kernelconv_hid4_3x3/biasconv_hid4_4x4/kernelconv_hid4_4x4/biasconv_hid5_5x5/kernelconv_hid5_5x5/biasoutput_C/kerneloutput_C/biasconv_rev0_3x3/kernelconv_rev0_3x3/biasconv_rev0_4x4/kernelconv_rev0_4x4/biasconv_rev0_5x5/kernelconv_rev0_5x5/biasconv_rev1_3x3/kernelconv_rev1_3x3/biasconv_rev1_4x4/kernelconv_rev1_4x4/biasconv_rev1_5x5/kernelconv_rev1_5x5/biasconv_rev2_3x3/kernelconv_rev2_3x3/biasconv_rev2_4x4/kernelconv_rev2_4x4/biasconv_rev2_5x5/kernelconv_rev2_5x5/biasconv_rev3_3x3/kernelconv_rev3_3x3/biasconv_rev3_4x4/kernelconv_rev3_4x4/biasconv_rev3_5x5/kernelconv_rev3_5x5/biasconv_rev4_3x3/kernelconv_rev4_3x3/biasconv_rev4_4x4/kernelconv_rev4_4x4/biasconv_rev5_5x5/kernelconv_rev5_5x5/biasoutput_S/kerneloutput_S/bias	iterationlearning_ratem/conv_prep0_3x3/kernelv/conv_prep0_3x3/kernelm/conv_prep0_3x3/biasv/conv_prep0_3x3/biasm/conv_prep0_4x4/kernelv/conv_prep0_4x4/kernelm/conv_prep0_4x4/biasv/conv_prep0_4x4/biasm/conv_prep0_5x5/kernelv/conv_prep0_5x5/kernelm/conv_prep0_5x5/biasv/conv_prep0_5x5/biasm/conv_prep1_3x3/kernelv/conv_prep1_3x3/kernelm/conv_prep1_3x3/biasv/conv_prep1_3x3/biasm/conv_prep1_4x4/kernelv/conv_prep1_4x4/kernelm/conv_prep1_4x4/biasv/conv_prep1_4x4/biasm/conv_prep1_5x5/kernelv/conv_prep1_5x5/kernelm/conv_prep1_5x5/biasv/conv_prep1_5x5/biasm/conv_hid0_3x3/kernelv/conv_hid0_3x3/kernelm/conv_hid0_3x3/biasv/conv_hid0_3x3/biasm/conv_hid0_4x4/kernelv/conv_hid0_4x4/kernelm/conv_hid0_4x4/biasv/conv_hid0_4x4/biasm/conv_hid0_5x5/kernelv/conv_hid0_5x5/kernelm/conv_hid0_5x5/biasv/conv_hid0_5x5/biasm/conv_hid1_3x3/kernelv/conv_hid1_3x3/kernelm/conv_hid1_3x3/biasv/conv_hid1_3x3/biasm/conv_hid1_4x4/kernelv/conv_hid1_4x4/kernelm/conv_hid1_4x4/biasv/conv_hid1_4x4/biasm/conv_hid1_5x5/kernelv/conv_hid1_5x5/kernelm/conv_hid1_5x5/biasv/conv_hid1_5x5/biasm/conv_hid2_3x3/kernelv/conv_hid2_3x3/kernelm/conv_hid2_3x3/biasv/conv_hid2_3x3/biasm/conv_hid2_4x4/kernelv/conv_hid2_4x4/kernelm/conv_hid2_4x4/biasv/conv_hid2_4x4/biasm/conv_hid2_5x5/kernelv/conv_hid2_5x5/kernelm/conv_hid2_5x5/biasv/conv_hid2_5x5/biasm/conv_hid3_3x3/kernelv/conv_hid3_3x3/kernelm/conv_hid3_3x3/biasv/conv_hid3_3x3/biasm/conv_hid3_4x4/kernelv/conv_hid3_4x4/kernelm/conv_hid3_4x4/biasv/conv_hid3_4x4/biasm/conv_hid3_5x5/kernelv/conv_hid3_5x5/kernelm/conv_hid3_5x5/biasv/conv_hid3_5x5/biasm/conv_hid4_3x3/kernelv/conv_hid4_3x3/kernelm/conv_hid4_3x3/biasv/conv_hid4_3x3/biasm/conv_hid4_4x4/kernelv/conv_hid4_4x4/kernelm/conv_hid4_4x4/biasv/conv_hid4_4x4/biasm/conv_hid5_5x5/kernelv/conv_hid5_5x5/kernelm/conv_hid5_5x5/biasv/conv_hid5_5x5/biasm/output_C/kernelv/output_C/kernelm/output_C/biasv/output_C/biastotalcountConst*�
Tin�
�2�*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference__traced_save_6833
� 
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_prep0_3x3/kernelconv_prep0_3x3/biasconv_prep0_4x4/kernelconv_prep0_4x4/biasconv_prep0_5x5/kernelconv_prep0_5x5/biasconv_prep1_3x3/kernelconv_prep1_3x3/biasconv_prep1_4x4/kernelconv_prep1_4x4/biasconv_prep1_5x5/kernelconv_prep1_5x5/biasconv_hid0_3x3/kernelconv_hid0_3x3/biasconv_hid0_4x4/kernelconv_hid0_4x4/biasconv_hid0_5x5/kernelconv_hid0_5x5/biasconv_hid1_3x3/kernelconv_hid1_3x3/biasconv_hid1_4x4/kernelconv_hid1_4x4/biasconv_hid1_5x5/kernelconv_hid1_5x5/biasconv_hid2_3x3/kernelconv_hid2_3x3/biasconv_hid2_4x4/kernelconv_hid2_4x4/biasconv_hid2_5x5/kernelconv_hid2_5x5/biasconv_hid3_3x3/kernelconv_hid3_3x3/biasconv_hid3_4x4/kernelconv_hid3_4x4/biasconv_hid3_5x5/kernelconv_hid3_5x5/biasconv_hid4_3x3/kernelconv_hid4_3x3/biasconv_hid4_4x4/kernelconv_hid4_4x4/biasconv_hid5_5x5/kernelconv_hid5_5x5/biasoutput_C/kerneloutput_C/biasconv_rev0_3x3/kernelconv_rev0_3x3/biasconv_rev0_4x4/kernelconv_rev0_4x4/biasconv_rev0_5x5/kernelconv_rev0_5x5/biasconv_rev1_3x3/kernelconv_rev1_3x3/biasconv_rev1_4x4/kernelconv_rev1_4x4/biasconv_rev1_5x5/kernelconv_rev1_5x5/biasconv_rev2_3x3/kernelconv_rev2_3x3/biasconv_rev2_4x4/kernelconv_rev2_4x4/biasconv_rev2_5x5/kernelconv_rev2_5x5/biasconv_rev3_3x3/kernelconv_rev3_3x3/biasconv_rev3_4x4/kernelconv_rev3_4x4/biasconv_rev3_5x5/kernelconv_rev3_5x5/biasconv_rev4_3x3/kernelconv_rev4_3x3/biasconv_rev4_4x4/kernelconv_rev4_4x4/biasconv_rev5_5x5/kernelconv_rev5_5x5/biasoutput_S/kerneloutput_S/bias	iterationlearning_ratem/conv_prep0_3x3/kernelv/conv_prep0_3x3/kernelm/conv_prep0_3x3/biasv/conv_prep0_3x3/biasm/conv_prep0_4x4/kernelv/conv_prep0_4x4/kernelm/conv_prep0_4x4/biasv/conv_prep0_4x4/biasm/conv_prep0_5x5/kernelv/conv_prep0_5x5/kernelm/conv_prep0_5x5/biasv/conv_prep0_5x5/biasm/conv_prep1_3x3/kernelv/conv_prep1_3x3/kernelm/conv_prep1_3x3/biasv/conv_prep1_3x3/biasm/conv_prep1_4x4/kernelv/conv_prep1_4x4/kernelm/conv_prep1_4x4/biasv/conv_prep1_4x4/biasm/conv_prep1_5x5/kernelv/conv_prep1_5x5/kernelm/conv_prep1_5x5/biasv/conv_prep1_5x5/biasm/conv_hid0_3x3/kernelv/conv_hid0_3x3/kernelm/conv_hid0_3x3/biasv/conv_hid0_3x3/biasm/conv_hid0_4x4/kernelv/conv_hid0_4x4/kernelm/conv_hid0_4x4/biasv/conv_hid0_4x4/biasm/conv_hid0_5x5/kernelv/conv_hid0_5x5/kernelm/conv_hid0_5x5/biasv/conv_hid0_5x5/biasm/conv_hid1_3x3/kernelv/conv_hid1_3x3/kernelm/conv_hid1_3x3/biasv/conv_hid1_3x3/biasm/conv_hid1_4x4/kernelv/conv_hid1_4x4/kernelm/conv_hid1_4x4/biasv/conv_hid1_4x4/biasm/conv_hid1_5x5/kernelv/conv_hid1_5x5/kernelm/conv_hid1_5x5/biasv/conv_hid1_5x5/biasm/conv_hid2_3x3/kernelv/conv_hid2_3x3/kernelm/conv_hid2_3x3/biasv/conv_hid2_3x3/biasm/conv_hid2_4x4/kernelv/conv_hid2_4x4/kernelm/conv_hid2_4x4/biasv/conv_hid2_4x4/biasm/conv_hid2_5x5/kernelv/conv_hid2_5x5/kernelm/conv_hid2_5x5/biasv/conv_hid2_5x5/biasm/conv_hid3_3x3/kernelv/conv_hid3_3x3/kernelm/conv_hid3_3x3/biasv/conv_hid3_3x3/biasm/conv_hid3_4x4/kernelv/conv_hid3_4x4/kernelm/conv_hid3_4x4/biasv/conv_hid3_4x4/biasm/conv_hid3_5x5/kernelv/conv_hid3_5x5/kernelm/conv_hid3_5x5/biasv/conv_hid3_5x5/biasm/conv_hid4_3x3/kernelv/conv_hid4_3x3/kernelm/conv_hid4_3x3/biasv/conv_hid4_3x3/biasm/conv_hid4_4x4/kernelv/conv_hid4_4x4/kernelm/conv_hid4_4x4/biasv/conv_hid4_4x4/biasm/conv_hid5_5x5/kernelv/conv_hid5_5x5/kernelm/conv_hid5_5x5/biasv/conv_hid5_5x5/biasm/output_C/kernelv/output_C/kernelm/output_C/biasv/output_C/biastotalcount*�
Tin�
�2�*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_restore_7346��'
�
�
,__inference_conv_hid0_3x3_layer_call_fn_4996

inputs!
unknown:D2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid0_3x3_layer_call_and_return_conditional_losses_2407w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@2<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@D: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name4992:$ 

_user_specified_name4990:W S
/
_output_shapes
:���������@@D
 
_user_specified_nameinputs
�
r
H__inference_concatenate_13_layer_call_and_return_conditional_losses_4165

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������@@:���������@@:WS
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
,__inference_conv_hid3_3x3_layer_call_fn_5221

inputs!
unknown:A2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid3_3x3_layer_call_and_return_conditional_losses_2578w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@2<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5217:$ 

_user_specified_name5215:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
f
,__inference_concatenate_5_layer_call_fn_5204
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_5_layer_call_and_return_conditional_losses_2566h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
�
,__inference_conv_hid2_5x5_layer_call_fn_5186

inputs!
unknown:A
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid2_5x5_layer_call_and_return_conditional_losses_2553w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5182:$ 

_user_specified_name5180:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_hid1_4x4_layer_call_and_return_conditional_losses_2480

inputs8
conv2d_readvariableop_resource:A
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_hid2_3x3_layer_call_and_return_conditional_losses_5157

inputs8
conv2d_readvariableop_resource:A2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_hid1_3x3_layer_call_and_return_conditional_losses_2464

inputs8
conv2d_readvariableop_resource:A2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
H__inference_concatenate_12_layer_call_and_return_conditional_losses_3561

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:WS
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������@@

 
_user_specified_nameinputs:W S
/
_output_shapes
:���������@@2
 
_user_specified_nameinputs
�
�
,__inference_conv_rev1_4x4_layer_call_fn_5511

inputs!
unknown:A

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev1_4x4_layer_call_and_return_conditional_losses_3361w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5507:$ 

_user_specified_name5505:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�

G__inference_concatenate_7_layer_call_and_return_conditional_losses_2680

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:WS
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������@@

 
_user_specified_nameinputs:W S
/
_output_shapes
:���������@@2
 
_user_specified_nameinputs
�
�
G__inference_conv_rev4_4x4_layer_call_and_return_conditional_losses_5747

inputs8
conv2d_readvariableop_resource:A
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
,__inference_conv_hid1_3x3_layer_call_fn_5071

inputs!
unknown:A2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid1_3x3_layer_call_and_return_conditional_losses_2464w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@2<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5067:$ 

_user_specified_name5065:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_rev2_3x3_layer_call_and_return_conditional_losses_5577

inputs8
conv2d_readvariableop_resource:A2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_hid3_3x3_layer_call_and_return_conditional_losses_2578

inputs8
conv2d_readvariableop_resource:A2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_hid1_3x3_layer_call_and_return_conditional_losses_5082

inputs8
conv2d_readvariableop_resource:A2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_hid3_4x4_layer_call_and_return_conditional_losses_2594

inputs8
conv2d_readvariableop_resource:A
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_rev5_5x5_layer_call_and_return_conditional_losses_3548

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
,__inference_conv_rev1_5x5_layer_call_fn_5531

inputs!
unknown:A
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev1_5x5_layer_call_and_return_conditional_losses_3377w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5527:$ 

_user_specified_name5525:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
H__inference_conv_prep1_3x3_layer_call_and_return_conditional_losses_2342

inputs8
conv2d_readvariableop_resource:A2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_hid0_3x3_layer_call_and_return_conditional_losses_2407

inputs8
conv2d_readvariableop_resource:D2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:D2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@D: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@D
 
_user_specified_nameinputs
�
�
G__inference_conv_rev3_5x5_layer_call_and_return_conditional_losses_5692

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
d
*__inference_concatenate_layer_call_fn_4891
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_2330h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
�
H__inference_concatenate_11_layer_call_and_return_conditional_losses_5707
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�	
g
H__inference_output_C_noise_layer_call_and_return_conditional_losses_5403

inputs
identity�I
ShapeShapeinputs*
T0*
_output_shapes
::��W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:���������@@*
dtype0�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:���������@@�
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:���������@@a
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:���������@@W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@@:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
,__inference_conv_hid1_5x5_layer_call_fn_5111

inputs!
unknown:A
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid1_5x5_layer_call_and_return_conditional_losses_2496w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5107:$ 

_user_specified_name5105:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_hid5_5x5_layer_call_and_return_conditional_losses_5347

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_hid3_5x5_layer_call_and_return_conditional_losses_5272

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
,__inference_conv_hid2_4x4_layer_call_fn_5166

inputs!
unknown:A

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid2_4x4_layer_call_and_return_conditional_losses_2537w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5162:$ 

_user_specified_name5160:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
f
,__inference_concatenate_6_layer_call_fn_5279
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_6_layer_call_and_return_conditional_losses_2623h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
Y
-__inference_concatenate_13_layer_call_fn_4817
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_concatenate_13_layer_call_and_return_conditional_losses_4165h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������@@:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_0
۠
�
A__inference_Encoder_layer_call_and_return_conditional_losses_2822
input_3
input_4-
conv_prep0_3x3_2703:2!
conv_prep0_3x3_2705:2-
conv_prep0_4x4_2708:
!
conv_prep0_4x4_2710:
-
conv_prep0_5x5_2713:!
conv_prep0_5x5_2715:-
conv_prep1_3x3_2719:A2!
conv_prep1_3x3_2721:2-
conv_prep1_4x4_2724:A
!
conv_prep1_4x4_2726:
-
conv_prep1_5x5_2729:A!
conv_prep1_5x5_2731:,
conv_hid0_3x3_2736:D2 
conv_hid0_3x3_2738:2,
conv_hid0_4x4_2741:D
 
conv_hid0_4x4_2743:
,
conv_hid0_5x5_2746:D 
conv_hid0_5x5_2748:,
conv_hid1_3x3_2752:A2 
conv_hid1_3x3_2754:2,
conv_hid1_4x4_2757:A
 
conv_hid1_4x4_2759:
,
conv_hid1_5x5_2762:A 
conv_hid1_5x5_2764:,
conv_hid2_3x3_2768:A2 
conv_hid2_3x3_2770:2,
conv_hid2_4x4_2773:A
 
conv_hid2_4x4_2775:
,
conv_hid2_5x5_2778:A 
conv_hid2_5x5_2780:,
conv_hid3_3x3_2784:A2 
conv_hid3_3x3_2786:2,
conv_hid3_4x4_2789:A
 
conv_hid3_4x4_2791:
,
conv_hid3_5x5_2794:A 
conv_hid3_5x5_2796:,
conv_hid4_3x3_2800:A2 
conv_hid4_3x3_2802:2,
conv_hid4_4x4_2805:A
 
conv_hid4_4x4_2807:
,
conv_hid5_5x5_2810:A 
conv_hid5_5x5_2812:'
output_c_2816:A
output_c_2818:
identity��%conv_hid0_3x3/StatefulPartitionedCall�%conv_hid0_4x4/StatefulPartitionedCall�%conv_hid0_5x5/StatefulPartitionedCall�%conv_hid1_3x3/StatefulPartitionedCall�%conv_hid1_4x4/StatefulPartitionedCall�%conv_hid1_5x5/StatefulPartitionedCall�%conv_hid2_3x3/StatefulPartitionedCall�%conv_hid2_4x4/StatefulPartitionedCall�%conv_hid2_5x5/StatefulPartitionedCall�%conv_hid3_3x3/StatefulPartitionedCall�%conv_hid3_4x4/StatefulPartitionedCall�%conv_hid3_5x5/StatefulPartitionedCall�%conv_hid4_3x3/StatefulPartitionedCall�%conv_hid4_4x4/StatefulPartitionedCall�%conv_hid5_5x5/StatefulPartitionedCall�&conv_prep0_3x3/StatefulPartitionedCall�&conv_prep0_4x4/StatefulPartitionedCall�&conv_prep0_5x5/StatefulPartitionedCall�&conv_prep1_3x3/StatefulPartitionedCall�&conv_prep1_4x4/StatefulPartitionedCall�&conv_prep1_5x5/StatefulPartitionedCall� output_C/StatefulPartitionedCall�
&conv_prep0_3x3/StatefulPartitionedCallStatefulPartitionedCallinput_3conv_prep0_3x3_2703conv_prep0_3x3_2705*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv_prep0_3x3_layer_call_and_return_conditional_losses_2285�
&conv_prep0_4x4/StatefulPartitionedCallStatefulPartitionedCallinput_3conv_prep0_4x4_2708conv_prep0_4x4_2710*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv_prep0_4x4_layer_call_and_return_conditional_losses_2301�
&conv_prep0_5x5/StatefulPartitionedCallStatefulPartitionedCallinput_3conv_prep0_5x5_2713conv_prep0_5x5_2715*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv_prep0_5x5_layer_call_and_return_conditional_losses_2317�
concatenate/PartitionedCallPartitionedCall/conv_prep0_3x3/StatefulPartitionedCall:output:0/conv_prep0_4x4/StatefulPartitionedCall:output:0/conv_prep0_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_2330�
&conv_prep1_3x3/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv_prep1_3x3_2719conv_prep1_3x3_2721*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv_prep1_3x3_layer_call_and_return_conditional_losses_2342�
&conv_prep1_4x4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv_prep1_4x4_2724conv_prep1_4x4_2726*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv_prep1_4x4_layer_call_and_return_conditional_losses_2358�
&conv_prep1_5x5/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv_prep1_5x5_2729conv_prep1_5x5_2731*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv_prep1_5x5_layer_call_and_return_conditional_losses_2374�
concatenate_1/PartitionedCallPartitionedCall/conv_prep1_3x3/StatefulPartitionedCall:output:0/conv_prep1_4x4/StatefulPartitionedCall:output:0/conv_prep1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_1_layer_call_and_return_conditional_losses_2387�
concatenate_2/PartitionedCallPartitionedCallinput_4&concatenate_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@D* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_2_layer_call_and_return_conditional_losses_2395�
%conv_hid0_3x3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv_hid0_3x3_2736conv_hid0_3x3_2738*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid0_3x3_layer_call_and_return_conditional_losses_2407�
%conv_hid0_4x4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv_hid0_4x4_2741conv_hid0_4x4_2743*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid0_4x4_layer_call_and_return_conditional_losses_2423�
%conv_hid0_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv_hid0_5x5_2746conv_hid0_5x5_2748*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid0_5x5_layer_call_and_return_conditional_losses_2439�
concatenate_3/PartitionedCallPartitionedCall.conv_hid0_3x3/StatefulPartitionedCall:output:0.conv_hid0_4x4/StatefulPartitionedCall:output:0.conv_hid0_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_3_layer_call_and_return_conditional_losses_2452�
%conv_hid1_3x3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv_hid1_3x3_2752conv_hid1_3x3_2754*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid1_3x3_layer_call_and_return_conditional_losses_2464�
%conv_hid1_4x4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv_hid1_4x4_2757conv_hid1_4x4_2759*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid1_4x4_layer_call_and_return_conditional_losses_2480�
%conv_hid1_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv_hid1_5x5_2762conv_hid1_5x5_2764*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid1_5x5_layer_call_and_return_conditional_losses_2496�
concatenate_4/PartitionedCallPartitionedCall.conv_hid1_3x3/StatefulPartitionedCall:output:0.conv_hid1_4x4/StatefulPartitionedCall:output:0.conv_hid1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_4_layer_call_and_return_conditional_losses_2509�
%conv_hid2_3x3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv_hid2_3x3_2768conv_hid2_3x3_2770*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid2_3x3_layer_call_and_return_conditional_losses_2521�
%conv_hid2_4x4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv_hid2_4x4_2773conv_hid2_4x4_2775*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid2_4x4_layer_call_and_return_conditional_losses_2537�
%conv_hid2_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv_hid2_5x5_2778conv_hid2_5x5_2780*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid2_5x5_layer_call_and_return_conditional_losses_2553�
concatenate_5/PartitionedCallPartitionedCall.conv_hid2_3x3/StatefulPartitionedCall:output:0.conv_hid2_4x4/StatefulPartitionedCall:output:0.conv_hid2_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_5_layer_call_and_return_conditional_losses_2566�
%conv_hid3_3x3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv_hid3_3x3_2784conv_hid3_3x3_2786*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid3_3x3_layer_call_and_return_conditional_losses_2578�
%conv_hid3_4x4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv_hid3_4x4_2789conv_hid3_4x4_2791*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid3_4x4_layer_call_and_return_conditional_losses_2594�
%conv_hid3_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv_hid3_5x5_2794conv_hid3_5x5_2796*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid3_5x5_layer_call_and_return_conditional_losses_2610�
concatenate_6/PartitionedCallPartitionedCall.conv_hid3_3x3/StatefulPartitionedCall:output:0.conv_hid3_4x4/StatefulPartitionedCall:output:0.conv_hid3_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_6_layer_call_and_return_conditional_losses_2623�
%conv_hid4_3x3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_6/PartitionedCall:output:0conv_hid4_3x3_2800conv_hid4_3x3_2802*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid4_3x3_layer_call_and_return_conditional_losses_2635�
%conv_hid4_4x4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_6/PartitionedCall:output:0conv_hid4_4x4_2805conv_hid4_4x4_2807*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid4_4x4_layer_call_and_return_conditional_losses_2651�
%conv_hid5_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_6/PartitionedCall:output:0conv_hid5_5x5_2810conv_hid5_5x5_2812*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid5_5x5_layer_call_and_return_conditional_losses_2667�
concatenate_7/PartitionedCallPartitionedCall.conv_hid4_3x3/StatefulPartitionedCall:output:0.conv_hid4_4x4/StatefulPartitionedCall:output:0.conv_hid5_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_7_layer_call_and_return_conditional_losses_2680�
 output_C/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0output_c_2816output_c_2818*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_output_C_layer_call_and_return_conditional_losses_2692�
IdentityIdentity)output_C/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@�
NoOpNoOp&^conv_hid0_3x3/StatefulPartitionedCall&^conv_hid0_4x4/StatefulPartitionedCall&^conv_hid0_5x5/StatefulPartitionedCall&^conv_hid1_3x3/StatefulPartitionedCall&^conv_hid1_4x4/StatefulPartitionedCall&^conv_hid1_5x5/StatefulPartitionedCall&^conv_hid2_3x3/StatefulPartitionedCall&^conv_hid2_4x4/StatefulPartitionedCall&^conv_hid2_5x5/StatefulPartitionedCall&^conv_hid3_3x3/StatefulPartitionedCall&^conv_hid3_4x4/StatefulPartitionedCall&^conv_hid3_5x5/StatefulPartitionedCall&^conv_hid4_3x3/StatefulPartitionedCall&^conv_hid4_4x4/StatefulPartitionedCall&^conv_hid5_5x5/StatefulPartitionedCall'^conv_prep0_3x3/StatefulPartitionedCall'^conv_prep0_4x4/StatefulPartitionedCall'^conv_prep0_5x5/StatefulPartitionedCall'^conv_prep1_3x3/StatefulPartitionedCall'^conv_prep1_4x4/StatefulPartitionedCall'^conv_prep1_5x5/StatefulPartitionedCall!^output_C/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@@:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%conv_hid0_3x3/StatefulPartitionedCall%conv_hid0_3x3/StatefulPartitionedCall2N
%conv_hid0_4x4/StatefulPartitionedCall%conv_hid0_4x4/StatefulPartitionedCall2N
%conv_hid0_5x5/StatefulPartitionedCall%conv_hid0_5x5/StatefulPartitionedCall2N
%conv_hid1_3x3/StatefulPartitionedCall%conv_hid1_3x3/StatefulPartitionedCall2N
%conv_hid1_4x4/StatefulPartitionedCall%conv_hid1_4x4/StatefulPartitionedCall2N
%conv_hid1_5x5/StatefulPartitionedCall%conv_hid1_5x5/StatefulPartitionedCall2N
%conv_hid2_3x3/StatefulPartitionedCall%conv_hid2_3x3/StatefulPartitionedCall2N
%conv_hid2_4x4/StatefulPartitionedCall%conv_hid2_4x4/StatefulPartitionedCall2N
%conv_hid2_5x5/StatefulPartitionedCall%conv_hid2_5x5/StatefulPartitionedCall2N
%conv_hid3_3x3/StatefulPartitionedCall%conv_hid3_3x3/StatefulPartitionedCall2N
%conv_hid3_4x4/StatefulPartitionedCall%conv_hid3_4x4/StatefulPartitionedCall2N
%conv_hid3_5x5/StatefulPartitionedCall%conv_hid3_5x5/StatefulPartitionedCall2N
%conv_hid4_3x3/StatefulPartitionedCall%conv_hid4_3x3/StatefulPartitionedCall2N
%conv_hid4_4x4/StatefulPartitionedCall%conv_hid4_4x4/StatefulPartitionedCall2N
%conv_hid5_5x5/StatefulPartitionedCall%conv_hid5_5x5/StatefulPartitionedCall2P
&conv_prep0_3x3/StatefulPartitionedCall&conv_prep0_3x3/StatefulPartitionedCall2P
&conv_prep0_4x4/StatefulPartitionedCall&conv_prep0_4x4/StatefulPartitionedCall2P
&conv_prep0_5x5/StatefulPartitionedCall&conv_prep0_5x5/StatefulPartitionedCall2P
&conv_prep1_3x3/StatefulPartitionedCall&conv_prep1_3x3/StatefulPartitionedCall2P
&conv_prep1_4x4/StatefulPartitionedCall&conv_prep1_4x4/StatefulPartitionedCall2P
&conv_prep1_5x5/StatefulPartitionedCall&conv_prep1_5x5/StatefulPartitionedCall2D
 output_C/StatefulPartitionedCall output_C/StatefulPartitionedCall:$- 

_user_specified_name2818:$, 

_user_specified_name2816:$+ 

_user_specified_name2812:$* 

_user_specified_name2810:$) 

_user_specified_name2807:$( 

_user_specified_name2805:$' 

_user_specified_name2802:$& 

_user_specified_name2800:$% 

_user_specified_name2796:$$ 

_user_specified_name2794:$# 

_user_specified_name2791:$" 

_user_specified_name2789:$! 

_user_specified_name2786:$  

_user_specified_name2784:$ 

_user_specified_name2780:$ 

_user_specified_name2778:$ 

_user_specified_name2775:$ 

_user_specified_name2773:$ 

_user_specified_name2770:$ 

_user_specified_name2768:$ 

_user_specified_name2764:$ 

_user_specified_name2762:$ 

_user_specified_name2759:$ 

_user_specified_name2757:$ 

_user_specified_name2754:$ 

_user_specified_name2752:$ 

_user_specified_name2748:$ 

_user_specified_name2746:$ 

_user_specified_name2743:$ 

_user_specified_name2741:$ 

_user_specified_name2738:$ 

_user_specified_name2736:$ 

_user_specified_name2731:$ 

_user_specified_name2729:$ 

_user_specified_name2726:$
 

_user_specified_name2724:$	 

_user_specified_name2721:$ 

_user_specified_name2719:$ 

_user_specified_name2715:$ 

_user_specified_name2713:$ 

_user_specified_name2710:$ 

_user_specified_name2708:$ 

_user_specified_name2705:$ 

_user_specified_name2703:XT
/
_output_shapes
:���������@@
!
_user_specified_name	input_4:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_3
�
�
&__inference_Decoder_layer_call_fn_3743
input_5!
unknown:2
	unknown_0:2#
	unknown_1:

	unknown_2:
#
	unknown_3:
	unknown_4:#
	unknown_5:A2
	unknown_6:2#
	unknown_7:A

	unknown_8:
#
	unknown_9:A

unknown_10:$

unknown_11:A2

unknown_12:2$

unknown_13:A


unknown_14:
$

unknown_15:A

unknown_16:$

unknown_17:A2

unknown_18:2$

unknown_19:A


unknown_20:
$

unknown_21:A

unknown_22:$

unknown_23:A2

unknown_24:2$

unknown_25:A


unknown_26:
$

unknown_27:A

unknown_28:$

unknown_29:A

unknown_30:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_Decoder_layer_call_and_return_conditional_losses_3580w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$  

_user_specified_name3739:$ 

_user_specified_name3737:$ 

_user_specified_name3735:$ 

_user_specified_name3733:$ 

_user_specified_name3731:$ 

_user_specified_name3729:$ 

_user_specified_name3727:$ 

_user_specified_name3725:$ 

_user_specified_name3723:$ 

_user_specified_name3721:$ 

_user_specified_name3719:$ 

_user_specified_name3717:$ 

_user_specified_name3715:$ 

_user_specified_name3713:$ 

_user_specified_name3711:$ 

_user_specified_name3709:$ 

_user_specified_name3707:$ 

_user_specified_name3705:$ 

_user_specified_name3703:$ 

_user_specified_name3701:$ 

_user_specified_name3699:$ 

_user_specified_name3697:$
 

_user_specified_name3695:$	 

_user_specified_name3693:$ 

_user_specified_name3691:$ 

_user_specified_name3689:$ 

_user_specified_name3687:$ 

_user_specified_name3685:$ 

_user_specified_name3683:$ 

_user_specified_name3681:$ 

_user_specified_name3679:$ 

_user_specified_name3677:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_5
�
�
G__inference_conv_rev2_4x4_layer_call_and_return_conditional_losses_3418

inputs8
conv2d_readvariableop_resource:A
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
,__inference_conv_rev0_5x5_layer_call_fn_5456

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev0_5x5_layer_call_and_return_conditional_losses_3320w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5452:$ 

_user_specified_name5450:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
G__inference_concatenate_5_layer_call_and_return_conditional_losses_5212
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
�
&__inference_Decoder_layer_call_fn_3812
input_5!
unknown:2
	unknown_0:2#
	unknown_1:

	unknown_2:
#
	unknown_3:
	unknown_4:#
	unknown_5:A2
	unknown_6:2#
	unknown_7:A

	unknown_8:
#
	unknown_9:A

unknown_10:$

unknown_11:A2

unknown_12:2$

unknown_13:A


unknown_14:
$

unknown_15:A

unknown_16:$

unknown_17:A2

unknown_18:2$

unknown_19:A


unknown_20:
$

unknown_21:A

unknown_22:$

unknown_23:A2

unknown_24:2$

unknown_25:A


unknown_26:
$

unknown_27:A

unknown_28:$

unknown_29:A

unknown_30:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_Decoder_layer_call_and_return_conditional_losses_3674w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$  

_user_specified_name3808:$ 

_user_specified_name3806:$ 

_user_specified_name3804:$ 

_user_specified_name3802:$ 

_user_specified_name3800:$ 

_user_specified_name3798:$ 

_user_specified_name3796:$ 

_user_specified_name3794:$ 

_user_specified_name3792:$ 

_user_specified_name3790:$ 

_user_specified_name3788:$ 

_user_specified_name3786:$ 

_user_specified_name3784:$ 

_user_specified_name3782:$ 

_user_specified_name3780:$ 

_user_specified_name3778:$ 

_user_specified_name3776:$ 

_user_specified_name3774:$ 

_user_specified_name3772:$ 

_user_specified_name3770:$ 

_user_specified_name3768:$ 

_user_specified_name3766:$
 

_user_specified_name3764:$	 

_user_specified_name3762:$ 

_user_specified_name3760:$ 

_user_specified_name3758:$ 

_user_specified_name3756:$ 

_user_specified_name3754:$ 

_user_specified_name3752:$ 

_user_specified_name3750:$ 

_user_specified_name3748:$ 

_user_specified_name3746:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_5
�
�
G__inference_conv_rev4_4x4_layer_call_and_return_conditional_losses_3532

inputs8
conv2d_readvariableop_resource:A
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
d
H__inference_output_C_noise_layer_call_and_return_conditional_losses_5407

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@@:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
G__inference_conv_rev1_4x4_layer_call_and_return_conditional_losses_5522

inputs8
conv2d_readvariableop_resource:A
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_rev4_3x3_layer_call_and_return_conditional_losses_3516

inputs8
conv2d_readvariableop_resource:A2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_hid4_3x3_layer_call_and_return_conditional_losses_5307

inputs8
conv2d_readvariableop_resource:A2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�	
g
H__inference_output_C_noise_layer_call_and_return_conditional_losses_3276

inputs
identity�I
ShapeShapeinputs*
T0*
_output_shapes
::��W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:���������@@*
dtype0�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:���������@@�
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:���������@@a
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:���������@@W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@@:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
G__inference_conv_rev1_3x3_layer_call_and_return_conditional_losses_5502

inputs8
conv2d_readvariableop_resource:A2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_rev3_5x5_layer_call_and_return_conditional_losses_3491

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
H__inference_conv_prep0_4x4_layer_call_and_return_conditional_losses_4864

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
f
,__inference_concatenate_8_layer_call_fn_5474
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_8_layer_call_and_return_conditional_losses_3333h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
�
G__inference_conv_rev0_5x5_layer_call_and_return_conditional_losses_5467

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
G__inference_conv_hid0_5x5_layer_call_and_return_conditional_losses_5047

inputs8
conv2d_readvariableop_resource:D-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:D*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@D: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@D
 
_user_specified_nameinputs
�
g
-__inference_concatenate_12_layer_call_fn_5774
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_concatenate_12_layer_call_and_return_conditional_losses_3561h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
�
,__inference_conv_rev1_3x3_layer_call_fn_5491

inputs!
unknown:A2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev1_3x3_layer_call_and_return_conditional_losses_3345w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@2<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5487:$ 

_user_specified_name5485:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_concatenate_9_layer_call_and_return_conditional_losses_5557
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
�
G__inference_conv_rev5_5x5_layer_call_and_return_conditional_losses_5767

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
,__inference_conv_rev0_3x3_layer_call_fn_5416

inputs!
unknown:2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev0_3x3_layer_call_and_return_conditional_losses_3288w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@2<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5412:$ 

_user_specified_name5410:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
f
,__inference_concatenate_7_layer_call_fn_5354
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_7_layer_call_and_return_conditional_losses_2680h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
�
G__inference_conv_hid3_5x5_layer_call_and_return_conditional_losses_2610

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
H__inference_conv_prep0_3x3_layer_call_and_return_conditional_losses_2285

inputs8
conv2d_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�

G__inference_concatenate_5_layer_call_and_return_conditional_losses_2566

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:WS
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������@@

 
_user_specified_nameinputs:W S
/
_output_shapes
:���������@@2
 
_user_specified_nameinputs
�
�
'__inference_output_S_layer_call_fn_5791

inputs!
unknown:A
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_output_S_layer_call_and_return_conditional_losses_3573w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5787:$ 

_user_specified_name5785:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
H__inference_conv_prep1_5x5_layer_call_and_return_conditional_losses_2374

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
,__inference_conv_hid3_5x5_layer_call_fn_5261

inputs!
unknown:A
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid3_5x5_layer_call_and_return_conditional_losses_2610w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5257:$ 

_user_specified_name5255:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_rev1_5x5_layer_call_and_return_conditional_losses_3377

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�

G__inference_concatenate_4_layer_call_and_return_conditional_losses_2509

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:WS
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������@@

 
_user_specified_nameinputs:W S
/
_output_shapes
:���������@@2
 
_user_specified_nameinputs
�
�
G__inference_concatenate_3_layer_call_and_return_conditional_losses_5062
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
�
-__inference_conv_prep0_3x3_layer_call_fn_4833

inputs!
unknown:2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv_prep0_3x3_layer_call_and_return_conditional_losses_2285w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@2<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name4829:$ 

_user_specified_name4827:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
f
,__inference_concatenate_1_layer_call_fn_4966
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_1_layer_call_and_return_conditional_losses_2387h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
�
G__inference_conv_hid1_4x4_layer_call_and_return_conditional_losses_5102

inputs8
conv2d_readvariableop_resource:A
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
f
,__inference_concatenate_4_layer_call_fn_5129
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_4_layer_call_and_return_conditional_losses_2509h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
�
H__inference_concatenate_11_layer_call_and_return_conditional_losses_3504

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:WS
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������@@

 
_user_specified_nameinputs:W S
/
_output_shapes
:���������@@2
 
_user_specified_nameinputs
�
�
G__inference_conv_rev0_4x4_layer_call_and_return_conditional_losses_3304

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
f
,__inference_concatenate_3_layer_call_fn_5054
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_3_layer_call_and_return_conditional_losses_2452h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
�
G__inference_conv_rev0_3x3_layer_call_and_return_conditional_losses_5427

inputs8
conv2d_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
G__inference_conv_hid4_4x4_layer_call_and_return_conditional_losses_5327

inputs8
conv2d_readvariableop_resource:A
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
��	
��
__inference__traced_save_6833
file_prefixF
,read_disablecopyonread_conv_prep0_3x3_kernel:2:
,read_1_disablecopyonread_conv_prep0_3x3_bias:2H
.read_2_disablecopyonread_conv_prep0_4x4_kernel:
:
,read_3_disablecopyonread_conv_prep0_4x4_bias:
H
.read_4_disablecopyonread_conv_prep0_5x5_kernel::
,read_5_disablecopyonread_conv_prep0_5x5_bias:H
.read_6_disablecopyonread_conv_prep1_3x3_kernel:A2:
,read_7_disablecopyonread_conv_prep1_3x3_bias:2H
.read_8_disablecopyonread_conv_prep1_4x4_kernel:A
:
,read_9_disablecopyonread_conv_prep1_4x4_bias:
I
/read_10_disablecopyonread_conv_prep1_5x5_kernel:A;
-read_11_disablecopyonread_conv_prep1_5x5_bias:H
.read_12_disablecopyonread_conv_hid0_3x3_kernel:D2:
,read_13_disablecopyonread_conv_hid0_3x3_bias:2H
.read_14_disablecopyonread_conv_hid0_4x4_kernel:D
:
,read_15_disablecopyonread_conv_hid0_4x4_bias:
H
.read_16_disablecopyonread_conv_hid0_5x5_kernel:D:
,read_17_disablecopyonread_conv_hid0_5x5_bias:H
.read_18_disablecopyonread_conv_hid1_3x3_kernel:A2:
,read_19_disablecopyonread_conv_hid1_3x3_bias:2H
.read_20_disablecopyonread_conv_hid1_4x4_kernel:A
:
,read_21_disablecopyonread_conv_hid1_4x4_bias:
H
.read_22_disablecopyonread_conv_hid1_5x5_kernel:A:
,read_23_disablecopyonread_conv_hid1_5x5_bias:H
.read_24_disablecopyonread_conv_hid2_3x3_kernel:A2:
,read_25_disablecopyonread_conv_hid2_3x3_bias:2H
.read_26_disablecopyonread_conv_hid2_4x4_kernel:A
:
,read_27_disablecopyonread_conv_hid2_4x4_bias:
H
.read_28_disablecopyonread_conv_hid2_5x5_kernel:A:
,read_29_disablecopyonread_conv_hid2_5x5_bias:H
.read_30_disablecopyonread_conv_hid3_3x3_kernel:A2:
,read_31_disablecopyonread_conv_hid3_3x3_bias:2H
.read_32_disablecopyonread_conv_hid3_4x4_kernel:A
:
,read_33_disablecopyonread_conv_hid3_4x4_bias:
H
.read_34_disablecopyonread_conv_hid3_5x5_kernel:A:
,read_35_disablecopyonread_conv_hid3_5x5_bias:H
.read_36_disablecopyonread_conv_hid4_3x3_kernel:A2:
,read_37_disablecopyonread_conv_hid4_3x3_bias:2H
.read_38_disablecopyonread_conv_hid4_4x4_kernel:A
:
,read_39_disablecopyonread_conv_hid4_4x4_bias:
H
.read_40_disablecopyonread_conv_hid5_5x5_kernel:A:
,read_41_disablecopyonread_conv_hid5_5x5_bias:C
)read_42_disablecopyonread_output_c_kernel:A5
'read_43_disablecopyonread_output_c_bias:H
.read_44_disablecopyonread_conv_rev0_3x3_kernel:2:
,read_45_disablecopyonread_conv_rev0_3x3_bias:2H
.read_46_disablecopyonread_conv_rev0_4x4_kernel:
:
,read_47_disablecopyonread_conv_rev0_4x4_bias:
H
.read_48_disablecopyonread_conv_rev0_5x5_kernel::
,read_49_disablecopyonread_conv_rev0_5x5_bias:H
.read_50_disablecopyonread_conv_rev1_3x3_kernel:A2:
,read_51_disablecopyonread_conv_rev1_3x3_bias:2H
.read_52_disablecopyonread_conv_rev1_4x4_kernel:A
:
,read_53_disablecopyonread_conv_rev1_4x4_bias:
H
.read_54_disablecopyonread_conv_rev1_5x5_kernel:A:
,read_55_disablecopyonread_conv_rev1_5x5_bias:H
.read_56_disablecopyonread_conv_rev2_3x3_kernel:A2:
,read_57_disablecopyonread_conv_rev2_3x3_bias:2H
.read_58_disablecopyonread_conv_rev2_4x4_kernel:A
:
,read_59_disablecopyonread_conv_rev2_4x4_bias:
H
.read_60_disablecopyonread_conv_rev2_5x5_kernel:A:
,read_61_disablecopyonread_conv_rev2_5x5_bias:H
.read_62_disablecopyonread_conv_rev3_3x3_kernel:A2:
,read_63_disablecopyonread_conv_rev3_3x3_bias:2H
.read_64_disablecopyonread_conv_rev3_4x4_kernel:A
:
,read_65_disablecopyonread_conv_rev3_4x4_bias:
H
.read_66_disablecopyonread_conv_rev3_5x5_kernel:A:
,read_67_disablecopyonread_conv_rev3_5x5_bias:H
.read_68_disablecopyonread_conv_rev4_3x3_kernel:A2:
,read_69_disablecopyonread_conv_rev4_3x3_bias:2H
.read_70_disablecopyonread_conv_rev4_4x4_kernel:A
:
,read_71_disablecopyonread_conv_rev4_4x4_bias:
H
.read_72_disablecopyonread_conv_rev5_5x5_kernel:A:
,read_73_disablecopyonread_conv_rev5_5x5_bias:C
)read_74_disablecopyonread_output_s_kernel:A5
'read_75_disablecopyonread_output_s_bias:-
#read_76_disablecopyonread_iteration:	 1
'read_77_disablecopyonread_learning_rate: K
1read_78_disablecopyonread_m_conv_prep0_3x3_kernel:2K
1read_79_disablecopyonread_v_conv_prep0_3x3_kernel:2=
/read_80_disablecopyonread_m_conv_prep0_3x3_bias:2=
/read_81_disablecopyonread_v_conv_prep0_3x3_bias:2K
1read_82_disablecopyonread_m_conv_prep0_4x4_kernel:
K
1read_83_disablecopyonread_v_conv_prep0_4x4_kernel:
=
/read_84_disablecopyonread_m_conv_prep0_4x4_bias:
=
/read_85_disablecopyonread_v_conv_prep0_4x4_bias:
K
1read_86_disablecopyonread_m_conv_prep0_5x5_kernel:K
1read_87_disablecopyonread_v_conv_prep0_5x5_kernel:=
/read_88_disablecopyonread_m_conv_prep0_5x5_bias:=
/read_89_disablecopyonread_v_conv_prep0_5x5_bias:K
1read_90_disablecopyonread_m_conv_prep1_3x3_kernel:A2K
1read_91_disablecopyonread_v_conv_prep1_3x3_kernel:A2=
/read_92_disablecopyonread_m_conv_prep1_3x3_bias:2=
/read_93_disablecopyonread_v_conv_prep1_3x3_bias:2K
1read_94_disablecopyonread_m_conv_prep1_4x4_kernel:A
K
1read_95_disablecopyonread_v_conv_prep1_4x4_kernel:A
=
/read_96_disablecopyonread_m_conv_prep1_4x4_bias:
=
/read_97_disablecopyonread_v_conv_prep1_4x4_bias:
K
1read_98_disablecopyonread_m_conv_prep1_5x5_kernel:AK
1read_99_disablecopyonread_v_conv_prep1_5x5_kernel:A>
0read_100_disablecopyonread_m_conv_prep1_5x5_bias:>
0read_101_disablecopyonread_v_conv_prep1_5x5_bias:K
1read_102_disablecopyonread_m_conv_hid0_3x3_kernel:D2K
1read_103_disablecopyonread_v_conv_hid0_3x3_kernel:D2=
/read_104_disablecopyonread_m_conv_hid0_3x3_bias:2=
/read_105_disablecopyonread_v_conv_hid0_3x3_bias:2K
1read_106_disablecopyonread_m_conv_hid0_4x4_kernel:D
K
1read_107_disablecopyonread_v_conv_hid0_4x4_kernel:D
=
/read_108_disablecopyonread_m_conv_hid0_4x4_bias:
=
/read_109_disablecopyonread_v_conv_hid0_4x4_bias:
K
1read_110_disablecopyonread_m_conv_hid0_5x5_kernel:DK
1read_111_disablecopyonread_v_conv_hid0_5x5_kernel:D=
/read_112_disablecopyonread_m_conv_hid0_5x5_bias:=
/read_113_disablecopyonread_v_conv_hid0_5x5_bias:K
1read_114_disablecopyonread_m_conv_hid1_3x3_kernel:A2K
1read_115_disablecopyonread_v_conv_hid1_3x3_kernel:A2=
/read_116_disablecopyonread_m_conv_hid1_3x3_bias:2=
/read_117_disablecopyonread_v_conv_hid1_3x3_bias:2K
1read_118_disablecopyonread_m_conv_hid1_4x4_kernel:A
K
1read_119_disablecopyonread_v_conv_hid1_4x4_kernel:A
=
/read_120_disablecopyonread_m_conv_hid1_4x4_bias:
=
/read_121_disablecopyonread_v_conv_hid1_4x4_bias:
K
1read_122_disablecopyonread_m_conv_hid1_5x5_kernel:AK
1read_123_disablecopyonread_v_conv_hid1_5x5_kernel:A=
/read_124_disablecopyonread_m_conv_hid1_5x5_bias:=
/read_125_disablecopyonread_v_conv_hid1_5x5_bias:K
1read_126_disablecopyonread_m_conv_hid2_3x3_kernel:A2K
1read_127_disablecopyonread_v_conv_hid2_3x3_kernel:A2=
/read_128_disablecopyonread_m_conv_hid2_3x3_bias:2=
/read_129_disablecopyonread_v_conv_hid2_3x3_bias:2K
1read_130_disablecopyonread_m_conv_hid2_4x4_kernel:A
K
1read_131_disablecopyonread_v_conv_hid2_4x4_kernel:A
=
/read_132_disablecopyonread_m_conv_hid2_4x4_bias:
=
/read_133_disablecopyonread_v_conv_hid2_4x4_bias:
K
1read_134_disablecopyonread_m_conv_hid2_5x5_kernel:AK
1read_135_disablecopyonread_v_conv_hid2_5x5_kernel:A=
/read_136_disablecopyonread_m_conv_hid2_5x5_bias:=
/read_137_disablecopyonread_v_conv_hid2_5x5_bias:K
1read_138_disablecopyonread_m_conv_hid3_3x3_kernel:A2K
1read_139_disablecopyonread_v_conv_hid3_3x3_kernel:A2=
/read_140_disablecopyonread_m_conv_hid3_3x3_bias:2=
/read_141_disablecopyonread_v_conv_hid3_3x3_bias:2K
1read_142_disablecopyonread_m_conv_hid3_4x4_kernel:A
K
1read_143_disablecopyonread_v_conv_hid3_4x4_kernel:A
=
/read_144_disablecopyonread_m_conv_hid3_4x4_bias:
=
/read_145_disablecopyonread_v_conv_hid3_4x4_bias:
K
1read_146_disablecopyonread_m_conv_hid3_5x5_kernel:AK
1read_147_disablecopyonread_v_conv_hid3_5x5_kernel:A=
/read_148_disablecopyonread_m_conv_hid3_5x5_bias:=
/read_149_disablecopyonread_v_conv_hid3_5x5_bias:K
1read_150_disablecopyonread_m_conv_hid4_3x3_kernel:A2K
1read_151_disablecopyonread_v_conv_hid4_3x3_kernel:A2=
/read_152_disablecopyonread_m_conv_hid4_3x3_bias:2=
/read_153_disablecopyonread_v_conv_hid4_3x3_bias:2K
1read_154_disablecopyonread_m_conv_hid4_4x4_kernel:A
K
1read_155_disablecopyonread_v_conv_hid4_4x4_kernel:A
=
/read_156_disablecopyonread_m_conv_hid4_4x4_bias:
=
/read_157_disablecopyonread_v_conv_hid4_4x4_bias:
K
1read_158_disablecopyonread_m_conv_hid5_5x5_kernel:AK
1read_159_disablecopyonread_v_conv_hid5_5x5_kernel:A=
/read_160_disablecopyonread_m_conv_hid5_5x5_bias:=
/read_161_disablecopyonread_v_conv_hid5_5x5_bias:F
,read_162_disablecopyonread_m_output_c_kernel:AF
,read_163_disablecopyonread_v_output_c_kernel:A8
*read_164_disablecopyonread_m_output_c_bias:8
*read_165_disablecopyonread_v_output_c_bias:*
 read_166_disablecopyonread_total: *
 read_167_disablecopyonread_count: 
savev2_const
identity_337��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_100/DisableCopyOnRead�Read_100/ReadVariableOp�Read_101/DisableCopyOnRead�Read_101/ReadVariableOp�Read_102/DisableCopyOnRead�Read_102/ReadVariableOp�Read_103/DisableCopyOnRead�Read_103/ReadVariableOp�Read_104/DisableCopyOnRead�Read_104/ReadVariableOp�Read_105/DisableCopyOnRead�Read_105/ReadVariableOp�Read_106/DisableCopyOnRead�Read_106/ReadVariableOp�Read_107/DisableCopyOnRead�Read_107/ReadVariableOp�Read_108/DisableCopyOnRead�Read_108/ReadVariableOp�Read_109/DisableCopyOnRead�Read_109/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_110/DisableCopyOnRead�Read_110/ReadVariableOp�Read_111/DisableCopyOnRead�Read_111/ReadVariableOp�Read_112/DisableCopyOnRead�Read_112/ReadVariableOp�Read_113/DisableCopyOnRead�Read_113/ReadVariableOp�Read_114/DisableCopyOnRead�Read_114/ReadVariableOp�Read_115/DisableCopyOnRead�Read_115/ReadVariableOp�Read_116/DisableCopyOnRead�Read_116/ReadVariableOp�Read_117/DisableCopyOnRead�Read_117/ReadVariableOp�Read_118/DisableCopyOnRead�Read_118/ReadVariableOp�Read_119/DisableCopyOnRead�Read_119/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_120/DisableCopyOnRead�Read_120/ReadVariableOp�Read_121/DisableCopyOnRead�Read_121/ReadVariableOp�Read_122/DisableCopyOnRead�Read_122/ReadVariableOp�Read_123/DisableCopyOnRead�Read_123/ReadVariableOp�Read_124/DisableCopyOnRead�Read_124/ReadVariableOp�Read_125/DisableCopyOnRead�Read_125/ReadVariableOp�Read_126/DisableCopyOnRead�Read_126/ReadVariableOp�Read_127/DisableCopyOnRead�Read_127/ReadVariableOp�Read_128/DisableCopyOnRead�Read_128/ReadVariableOp�Read_129/DisableCopyOnRead�Read_129/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_130/DisableCopyOnRead�Read_130/ReadVariableOp�Read_131/DisableCopyOnRead�Read_131/ReadVariableOp�Read_132/DisableCopyOnRead�Read_132/ReadVariableOp�Read_133/DisableCopyOnRead�Read_133/ReadVariableOp�Read_134/DisableCopyOnRead�Read_134/ReadVariableOp�Read_135/DisableCopyOnRead�Read_135/ReadVariableOp�Read_136/DisableCopyOnRead�Read_136/ReadVariableOp�Read_137/DisableCopyOnRead�Read_137/ReadVariableOp�Read_138/DisableCopyOnRead�Read_138/ReadVariableOp�Read_139/DisableCopyOnRead�Read_139/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_140/DisableCopyOnRead�Read_140/ReadVariableOp�Read_141/DisableCopyOnRead�Read_141/ReadVariableOp�Read_142/DisableCopyOnRead�Read_142/ReadVariableOp�Read_143/DisableCopyOnRead�Read_143/ReadVariableOp�Read_144/DisableCopyOnRead�Read_144/ReadVariableOp�Read_145/DisableCopyOnRead�Read_145/ReadVariableOp�Read_146/DisableCopyOnRead�Read_146/ReadVariableOp�Read_147/DisableCopyOnRead�Read_147/ReadVariableOp�Read_148/DisableCopyOnRead�Read_148/ReadVariableOp�Read_149/DisableCopyOnRead�Read_149/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_150/DisableCopyOnRead�Read_150/ReadVariableOp�Read_151/DisableCopyOnRead�Read_151/ReadVariableOp�Read_152/DisableCopyOnRead�Read_152/ReadVariableOp�Read_153/DisableCopyOnRead�Read_153/ReadVariableOp�Read_154/DisableCopyOnRead�Read_154/ReadVariableOp�Read_155/DisableCopyOnRead�Read_155/ReadVariableOp�Read_156/DisableCopyOnRead�Read_156/ReadVariableOp�Read_157/DisableCopyOnRead�Read_157/ReadVariableOp�Read_158/DisableCopyOnRead�Read_158/ReadVariableOp�Read_159/DisableCopyOnRead�Read_159/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_160/DisableCopyOnRead�Read_160/ReadVariableOp�Read_161/DisableCopyOnRead�Read_161/ReadVariableOp�Read_162/DisableCopyOnRead�Read_162/ReadVariableOp�Read_163/DisableCopyOnRead�Read_163/ReadVariableOp�Read_164/DisableCopyOnRead�Read_164/ReadVariableOp�Read_165/DisableCopyOnRead�Read_165/ReadVariableOp�Read_166/DisableCopyOnRead�Read_166/ReadVariableOp�Read_167/DisableCopyOnRead�Read_167/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_69/DisableCopyOnRead�Read_69/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_70/DisableCopyOnRead�Read_70/ReadVariableOp�Read_71/DisableCopyOnRead�Read_71/ReadVariableOp�Read_72/DisableCopyOnRead�Read_72/ReadVariableOp�Read_73/DisableCopyOnRead�Read_73/ReadVariableOp�Read_74/DisableCopyOnRead�Read_74/ReadVariableOp�Read_75/DisableCopyOnRead�Read_75/ReadVariableOp�Read_76/DisableCopyOnRead�Read_76/ReadVariableOp�Read_77/DisableCopyOnRead�Read_77/ReadVariableOp�Read_78/DisableCopyOnRead�Read_78/ReadVariableOp�Read_79/DisableCopyOnRead�Read_79/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_80/DisableCopyOnRead�Read_80/ReadVariableOp�Read_81/DisableCopyOnRead�Read_81/ReadVariableOp�Read_82/DisableCopyOnRead�Read_82/ReadVariableOp�Read_83/DisableCopyOnRead�Read_83/ReadVariableOp�Read_84/DisableCopyOnRead�Read_84/ReadVariableOp�Read_85/DisableCopyOnRead�Read_85/ReadVariableOp�Read_86/DisableCopyOnRead�Read_86/ReadVariableOp�Read_87/DisableCopyOnRead�Read_87/ReadVariableOp�Read_88/DisableCopyOnRead�Read_88/ReadVariableOp�Read_89/DisableCopyOnRead�Read_89/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOp�Read_90/DisableCopyOnRead�Read_90/ReadVariableOp�Read_91/DisableCopyOnRead�Read_91/ReadVariableOp�Read_92/DisableCopyOnRead�Read_92/ReadVariableOp�Read_93/DisableCopyOnRead�Read_93/ReadVariableOp�Read_94/DisableCopyOnRead�Read_94/ReadVariableOp�Read_95/DisableCopyOnRead�Read_95/ReadVariableOp�Read_96/DisableCopyOnRead�Read_96/ReadVariableOp�Read_97/DisableCopyOnRead�Read_97/ReadVariableOp�Read_98/DisableCopyOnRead�Read_98/ReadVariableOp�Read_99/DisableCopyOnRead�Read_99/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ~
Read/DisableCopyOnReadDisableCopyOnRead,read_disablecopyonread_conv_prep0_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp,read_disablecopyonread_conv_prep0_3x3_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:2*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:2i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:2�
Read_1/DisableCopyOnReadDisableCopyOnRead,read_1_disablecopyonread_conv_prep0_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp,read_1_disablecopyonread_conv_prep0_3x3_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_2/DisableCopyOnReadDisableCopyOnRead.read_2_disablecopyonread_conv_prep0_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp.read_2_disablecopyonread_conv_prep0_4x4_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:
*
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:
k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
:
�
Read_3/DisableCopyOnReadDisableCopyOnRead,read_3_disablecopyonread_conv_prep0_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp,read_3_disablecopyonread_conv_prep0_4x4_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_4/DisableCopyOnReadDisableCopyOnRead.read_4_disablecopyonread_conv_prep0_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp.read_4_disablecopyonread_conv_prep0_5x5_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_5/DisableCopyOnReadDisableCopyOnRead,read_5_disablecopyonread_conv_prep0_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp,read_5_disablecopyonread_conv_prep0_5x5_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_6/DisableCopyOnReadDisableCopyOnRead.read_6_disablecopyonread_conv_prep1_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp.read_6_disablecopyonread_conv_prep1_3x3_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A2*
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A2m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
:A2�
Read_7/DisableCopyOnReadDisableCopyOnRead,read_7_disablecopyonread_conv_prep1_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp,read_7_disablecopyonread_conv_prep1_3x3_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_8/DisableCopyOnReadDisableCopyOnRead.read_8_disablecopyonread_conv_prep1_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp.read_8_disablecopyonread_conv_prep1_4x4_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A
*
dtype0v
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A
m
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*&
_output_shapes
:A
�
Read_9/DisableCopyOnReadDisableCopyOnRead,read_9_disablecopyonread_conv_prep1_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp,read_9_disablecopyonread_conv_prep1_4x4_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_10/DisableCopyOnReadDisableCopyOnRead/read_10_disablecopyonread_conv_prep1_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp/read_10_disablecopyonread_conv_prep1_5x5_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0w
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Am
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_11/DisableCopyOnReadDisableCopyOnRead-read_11_disablecopyonread_conv_prep1_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp-read_11_disablecopyonread_conv_prep1_5x5_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_12/DisableCopyOnReadDisableCopyOnRead.read_12_disablecopyonread_conv_hid0_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp.read_12_disablecopyonread_conv_hid0_3x3_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:D2*
dtype0w
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:D2m
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*&
_output_shapes
:D2�
Read_13/DisableCopyOnReadDisableCopyOnRead,read_13_disablecopyonread_conv_hid0_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp,read_13_disablecopyonread_conv_hid0_3x3_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_14/DisableCopyOnReadDisableCopyOnRead.read_14_disablecopyonread_conv_hid0_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp.read_14_disablecopyonread_conv_hid0_4x4_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:D
*
dtype0w
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:D
m
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*&
_output_shapes
:D
�
Read_15/DisableCopyOnReadDisableCopyOnRead,read_15_disablecopyonread_conv_hid0_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp,read_15_disablecopyonread_conv_hid0_4x4_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_16/DisableCopyOnReadDisableCopyOnRead.read_16_disablecopyonread_conv_hid0_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp.read_16_disablecopyonread_conv_hid0_5x5_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:D*
dtype0w
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Dm
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*&
_output_shapes
:D�
Read_17/DisableCopyOnReadDisableCopyOnRead,read_17_disablecopyonread_conv_hid0_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp,read_17_disablecopyonread_conv_hid0_5x5_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_18/DisableCopyOnReadDisableCopyOnRead.read_18_disablecopyonread_conv_hid1_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp.read_18_disablecopyonread_conv_hid1_3x3_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A2*
dtype0w
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A2m
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*&
_output_shapes
:A2�
Read_19/DisableCopyOnReadDisableCopyOnRead,read_19_disablecopyonread_conv_hid1_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp,read_19_disablecopyonread_conv_hid1_3x3_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_20/DisableCopyOnReadDisableCopyOnRead.read_20_disablecopyonread_conv_hid1_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp.read_20_disablecopyonread_conv_hid1_4x4_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A
*
dtype0w
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A
m
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*&
_output_shapes
:A
�
Read_21/DisableCopyOnReadDisableCopyOnRead,read_21_disablecopyonread_conv_hid1_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp,read_21_disablecopyonread_conv_hid1_4x4_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_22/DisableCopyOnReadDisableCopyOnRead.read_22_disablecopyonread_conv_hid1_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp.read_22_disablecopyonread_conv_hid1_5x5_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0w
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Am
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_23/DisableCopyOnReadDisableCopyOnRead,read_23_disablecopyonread_conv_hid1_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp,read_23_disablecopyonread_conv_hid1_5x5_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_24/DisableCopyOnReadDisableCopyOnRead.read_24_disablecopyonread_conv_hid2_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp.read_24_disablecopyonread_conv_hid2_3x3_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A2*
dtype0w
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A2m
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*&
_output_shapes
:A2�
Read_25/DisableCopyOnReadDisableCopyOnRead,read_25_disablecopyonread_conv_hid2_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp,read_25_disablecopyonread_conv_hid2_3x3_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_26/DisableCopyOnReadDisableCopyOnRead.read_26_disablecopyonread_conv_hid2_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp.read_26_disablecopyonread_conv_hid2_4x4_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A
*
dtype0w
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A
m
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*&
_output_shapes
:A
�
Read_27/DisableCopyOnReadDisableCopyOnRead,read_27_disablecopyonread_conv_hid2_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp,read_27_disablecopyonread_conv_hid2_4x4_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_28/DisableCopyOnReadDisableCopyOnRead.read_28_disablecopyonread_conv_hid2_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp.read_28_disablecopyonread_conv_hid2_5x5_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0w
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Am
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_29/DisableCopyOnReadDisableCopyOnRead,read_29_disablecopyonread_conv_hid2_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp,read_29_disablecopyonread_conv_hid2_5x5_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_30/DisableCopyOnReadDisableCopyOnRead.read_30_disablecopyonread_conv_hid3_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp.read_30_disablecopyonread_conv_hid3_3x3_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A2*
dtype0w
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A2m
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*&
_output_shapes
:A2�
Read_31/DisableCopyOnReadDisableCopyOnRead,read_31_disablecopyonread_conv_hid3_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp,read_31_disablecopyonread_conv_hid3_3x3_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_32/DisableCopyOnReadDisableCopyOnRead.read_32_disablecopyonread_conv_hid3_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp.read_32_disablecopyonread_conv_hid3_4x4_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A
*
dtype0w
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A
m
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*&
_output_shapes
:A
�
Read_33/DisableCopyOnReadDisableCopyOnRead,read_33_disablecopyonread_conv_hid3_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp,read_33_disablecopyonread_conv_hid3_4x4_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_34/DisableCopyOnReadDisableCopyOnRead.read_34_disablecopyonread_conv_hid3_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp.read_34_disablecopyonread_conv_hid3_5x5_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0w
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Am
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_35/DisableCopyOnReadDisableCopyOnRead,read_35_disablecopyonread_conv_hid3_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp,read_35_disablecopyonread_conv_hid3_5x5_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_36/DisableCopyOnReadDisableCopyOnRead.read_36_disablecopyonread_conv_hid4_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp.read_36_disablecopyonread_conv_hid4_3x3_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A2*
dtype0w
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A2m
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*&
_output_shapes
:A2�
Read_37/DisableCopyOnReadDisableCopyOnRead,read_37_disablecopyonread_conv_hid4_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp,read_37_disablecopyonread_conv_hid4_3x3_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_38/DisableCopyOnReadDisableCopyOnRead.read_38_disablecopyonread_conv_hid4_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp.read_38_disablecopyonread_conv_hid4_4x4_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A
*
dtype0w
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A
m
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*&
_output_shapes
:A
�
Read_39/DisableCopyOnReadDisableCopyOnRead,read_39_disablecopyonread_conv_hid4_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp,read_39_disablecopyonread_conv_hid4_4x4_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_40/DisableCopyOnReadDisableCopyOnRead.read_40_disablecopyonread_conv_hid5_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp.read_40_disablecopyonread_conv_hid5_5x5_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0w
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Am
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_41/DisableCopyOnReadDisableCopyOnRead,read_41_disablecopyonread_conv_hid5_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp,read_41_disablecopyonread_conv_hid5_5x5_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_42/DisableCopyOnReadDisableCopyOnRead)read_42_disablecopyonread_output_c_kernel"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp)read_42_disablecopyonread_output_c_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0w
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Am
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*&
_output_shapes
:A|
Read_43/DisableCopyOnReadDisableCopyOnRead'read_43_disablecopyonread_output_c_bias"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp'read_43_disablecopyonread_output_c_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_44/DisableCopyOnReadDisableCopyOnRead.read_44_disablecopyonread_conv_rev0_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp.read_44_disablecopyonread_conv_rev0_3x3_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:2*
dtype0w
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:2m
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*&
_output_shapes
:2�
Read_45/DisableCopyOnReadDisableCopyOnRead,read_45_disablecopyonread_conv_rev0_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp,read_45_disablecopyonread_conv_rev0_3x3_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_46/DisableCopyOnReadDisableCopyOnRead.read_46_disablecopyonread_conv_rev0_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp.read_46_disablecopyonread_conv_rev0_4x4_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:
*
dtype0w
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:
m
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*&
_output_shapes
:
�
Read_47/DisableCopyOnReadDisableCopyOnRead,read_47_disablecopyonread_conv_rev0_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp,read_47_disablecopyonread_conv_rev0_4x4_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_48/DisableCopyOnReadDisableCopyOnRead.read_48_disablecopyonread_conv_rev0_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp.read_48_disablecopyonread_conv_rev0_5x5_kernel^Read_48/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_49/DisableCopyOnReadDisableCopyOnRead,read_49_disablecopyonread_conv_rev0_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp,read_49_disablecopyonread_conv_rev0_5x5_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_50/DisableCopyOnReadDisableCopyOnRead.read_50_disablecopyonread_conv_rev1_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp.read_50_disablecopyonread_conv_rev1_3x3_kernel^Read_50/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A2*
dtype0x
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A2o
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*&
_output_shapes
:A2�
Read_51/DisableCopyOnReadDisableCopyOnRead,read_51_disablecopyonread_conv_rev1_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp,read_51_disablecopyonread_conv_rev1_3x3_bias^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0l
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2c
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_52/DisableCopyOnReadDisableCopyOnRead.read_52_disablecopyonread_conv_rev1_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp.read_52_disablecopyonread_conv_rev1_4x4_kernel^Read_52/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A
*
dtype0x
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A
o
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*&
_output_shapes
:A
�
Read_53/DisableCopyOnReadDisableCopyOnRead,read_53_disablecopyonread_conv_rev1_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp,read_53_disablecopyonread_conv_rev1_4x4_bias^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0l
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_54/DisableCopyOnReadDisableCopyOnRead.read_54_disablecopyonread_conv_rev1_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp.read_54_disablecopyonread_conv_rev1_5x5_kernel^Read_54/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0x
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Ao
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_55/DisableCopyOnReadDisableCopyOnRead,read_55_disablecopyonread_conv_rev1_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp,read_55_disablecopyonread_conv_rev1_5x5_bias^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_56/DisableCopyOnReadDisableCopyOnRead.read_56_disablecopyonread_conv_rev2_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp.read_56_disablecopyonread_conv_rev2_3x3_kernel^Read_56/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A2*
dtype0x
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A2o
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*&
_output_shapes
:A2�
Read_57/DisableCopyOnReadDisableCopyOnRead,read_57_disablecopyonread_conv_rev2_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp,read_57_disablecopyonread_conv_rev2_3x3_bias^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0l
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2c
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_58/DisableCopyOnReadDisableCopyOnRead.read_58_disablecopyonread_conv_rev2_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp.read_58_disablecopyonread_conv_rev2_4x4_kernel^Read_58/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A
*
dtype0x
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A
o
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*&
_output_shapes
:A
�
Read_59/DisableCopyOnReadDisableCopyOnRead,read_59_disablecopyonread_conv_rev2_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOp,read_59_disablecopyonread_conv_rev2_4x4_bias^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0l
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_60/DisableCopyOnReadDisableCopyOnRead.read_60_disablecopyonread_conv_rev2_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp.read_60_disablecopyonread_conv_rev2_5x5_kernel^Read_60/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0x
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Ao
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_61/DisableCopyOnReadDisableCopyOnRead,read_61_disablecopyonread_conv_rev2_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp,read_61_disablecopyonread_conv_rev2_5x5_bias^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_62/DisableCopyOnReadDisableCopyOnRead.read_62_disablecopyonread_conv_rev3_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp.read_62_disablecopyonread_conv_rev3_3x3_kernel^Read_62/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A2*
dtype0x
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A2o
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*&
_output_shapes
:A2�
Read_63/DisableCopyOnReadDisableCopyOnRead,read_63_disablecopyonread_conv_rev3_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp,read_63_disablecopyonread_conv_rev3_3x3_bias^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0l
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2c
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_64/DisableCopyOnReadDisableCopyOnRead.read_64_disablecopyonread_conv_rev3_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp.read_64_disablecopyonread_conv_rev3_4x4_kernel^Read_64/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A
*
dtype0x
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A
o
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*&
_output_shapes
:A
�
Read_65/DisableCopyOnReadDisableCopyOnRead,read_65_disablecopyonread_conv_rev3_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp,read_65_disablecopyonread_conv_rev3_4x4_bias^Read_65/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0l
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_66/DisableCopyOnReadDisableCopyOnRead.read_66_disablecopyonread_conv_rev3_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOp.read_66_disablecopyonread_conv_rev3_5x5_kernel^Read_66/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0x
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Ao
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_67/DisableCopyOnReadDisableCopyOnRead,read_67_disablecopyonread_conv_rev3_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOp,read_67_disablecopyonread_conv_rev3_5x5_bias^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_68/DisableCopyOnReadDisableCopyOnRead.read_68_disablecopyonread_conv_rev4_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOp.read_68_disablecopyonread_conv_rev4_3x3_kernel^Read_68/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A2*
dtype0x
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A2o
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*&
_output_shapes
:A2�
Read_69/DisableCopyOnReadDisableCopyOnRead,read_69_disablecopyonread_conv_rev4_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOp,read_69_disablecopyonread_conv_rev4_3x3_bias^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0l
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2c
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_70/DisableCopyOnReadDisableCopyOnRead.read_70_disablecopyonread_conv_rev4_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_70/ReadVariableOpReadVariableOp.read_70_disablecopyonread_conv_rev4_4x4_kernel^Read_70/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A
*
dtype0x
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A
o
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*&
_output_shapes
:A
�
Read_71/DisableCopyOnReadDisableCopyOnRead,read_71_disablecopyonread_conv_rev4_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_71/ReadVariableOpReadVariableOp,read_71_disablecopyonread_conv_rev4_4x4_bias^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0l
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_72/DisableCopyOnReadDisableCopyOnRead.read_72_disablecopyonread_conv_rev5_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_72/ReadVariableOpReadVariableOp.read_72_disablecopyonread_conv_rev5_5x5_kernel^Read_72/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0x
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Ao
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_73/DisableCopyOnReadDisableCopyOnRead,read_73_disablecopyonread_conv_rev5_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_73/ReadVariableOpReadVariableOp,read_73_disablecopyonread_conv_rev5_5x5_bias^Read_73/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_74/DisableCopyOnReadDisableCopyOnRead)read_74_disablecopyonread_output_s_kernel"/device:CPU:0*
_output_shapes
 �
Read_74/ReadVariableOpReadVariableOp)read_74_disablecopyonread_output_s_kernel^Read_74/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0x
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Ao
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*&
_output_shapes
:A|
Read_75/DisableCopyOnReadDisableCopyOnRead'read_75_disablecopyonread_output_s_bias"/device:CPU:0*
_output_shapes
 �
Read_75/ReadVariableOpReadVariableOp'read_75_disablecopyonread_output_s_bias^Read_75/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_76/DisableCopyOnReadDisableCopyOnRead#read_76_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_76/ReadVariableOpReadVariableOp#read_76_disablecopyonread_iteration^Read_76/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	h
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: _
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_77/DisableCopyOnReadDisableCopyOnRead'read_77_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_77/ReadVariableOpReadVariableOp'read_77_disablecopyonread_learning_rate^Read_77/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_78/DisableCopyOnReadDisableCopyOnRead1read_78_disablecopyonread_m_conv_prep0_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_78/ReadVariableOpReadVariableOp1read_78_disablecopyonread_m_conv_prep0_3x3_kernel^Read_78/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:2*
dtype0x
Identity_156IdentityRead_78/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:2o
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*&
_output_shapes
:2�
Read_79/DisableCopyOnReadDisableCopyOnRead1read_79_disablecopyonread_v_conv_prep0_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_79/ReadVariableOpReadVariableOp1read_79_disablecopyonread_v_conv_prep0_3x3_kernel^Read_79/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:2*
dtype0x
Identity_158IdentityRead_79/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:2o
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*&
_output_shapes
:2�
Read_80/DisableCopyOnReadDisableCopyOnRead/read_80_disablecopyonread_m_conv_prep0_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_80/ReadVariableOpReadVariableOp/read_80_disablecopyonread_m_conv_prep0_3x3_bias^Read_80/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0l
Identity_160IdentityRead_80/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2c
Identity_161IdentityIdentity_160:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_81/DisableCopyOnReadDisableCopyOnRead/read_81_disablecopyonread_v_conv_prep0_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_81/ReadVariableOpReadVariableOp/read_81_disablecopyonread_v_conv_prep0_3x3_bias^Read_81/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0l
Identity_162IdentityRead_81/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2c
Identity_163IdentityIdentity_162:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_82/DisableCopyOnReadDisableCopyOnRead1read_82_disablecopyonread_m_conv_prep0_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_82/ReadVariableOpReadVariableOp1read_82_disablecopyonread_m_conv_prep0_4x4_kernel^Read_82/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:
*
dtype0x
Identity_164IdentityRead_82/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:
o
Identity_165IdentityIdentity_164:output:0"/device:CPU:0*
T0*&
_output_shapes
:
�
Read_83/DisableCopyOnReadDisableCopyOnRead1read_83_disablecopyonread_v_conv_prep0_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_83/ReadVariableOpReadVariableOp1read_83_disablecopyonread_v_conv_prep0_4x4_kernel^Read_83/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:
*
dtype0x
Identity_166IdentityRead_83/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:
o
Identity_167IdentityIdentity_166:output:0"/device:CPU:0*
T0*&
_output_shapes
:
�
Read_84/DisableCopyOnReadDisableCopyOnRead/read_84_disablecopyonread_m_conv_prep0_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_84/ReadVariableOpReadVariableOp/read_84_disablecopyonread_m_conv_prep0_4x4_bias^Read_84/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0l
Identity_168IdentityRead_84/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_169IdentityIdentity_168:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_85/DisableCopyOnReadDisableCopyOnRead/read_85_disablecopyonread_v_conv_prep0_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_85/ReadVariableOpReadVariableOp/read_85_disablecopyonread_v_conv_prep0_4x4_bias^Read_85/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0l
Identity_170IdentityRead_85/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_171IdentityIdentity_170:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_86/DisableCopyOnReadDisableCopyOnRead1read_86_disablecopyonread_m_conv_prep0_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_86/ReadVariableOpReadVariableOp1read_86_disablecopyonread_m_conv_prep0_5x5_kernel^Read_86/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_172IdentityRead_86/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_173IdentityIdentity_172:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_87/DisableCopyOnReadDisableCopyOnRead1read_87_disablecopyonread_v_conv_prep0_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_87/ReadVariableOpReadVariableOp1read_87_disablecopyonread_v_conv_prep0_5x5_kernel^Read_87/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_174IdentityRead_87/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_175IdentityIdentity_174:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_88/DisableCopyOnReadDisableCopyOnRead/read_88_disablecopyonread_m_conv_prep0_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_88/ReadVariableOpReadVariableOp/read_88_disablecopyonread_m_conv_prep0_5x5_bias^Read_88/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_176IdentityRead_88/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_177IdentityIdentity_176:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_89/DisableCopyOnReadDisableCopyOnRead/read_89_disablecopyonread_v_conv_prep0_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_89/ReadVariableOpReadVariableOp/read_89_disablecopyonread_v_conv_prep0_5x5_bias^Read_89/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_178IdentityRead_89/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_179IdentityIdentity_178:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_90/DisableCopyOnReadDisableCopyOnRead1read_90_disablecopyonread_m_conv_prep1_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_90/ReadVariableOpReadVariableOp1read_90_disablecopyonread_m_conv_prep1_3x3_kernel^Read_90/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A2*
dtype0x
Identity_180IdentityRead_90/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A2o
Identity_181IdentityIdentity_180:output:0"/device:CPU:0*
T0*&
_output_shapes
:A2�
Read_91/DisableCopyOnReadDisableCopyOnRead1read_91_disablecopyonread_v_conv_prep1_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_91/ReadVariableOpReadVariableOp1read_91_disablecopyonread_v_conv_prep1_3x3_kernel^Read_91/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A2*
dtype0x
Identity_182IdentityRead_91/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A2o
Identity_183IdentityIdentity_182:output:0"/device:CPU:0*
T0*&
_output_shapes
:A2�
Read_92/DisableCopyOnReadDisableCopyOnRead/read_92_disablecopyonread_m_conv_prep1_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_92/ReadVariableOpReadVariableOp/read_92_disablecopyonread_m_conv_prep1_3x3_bias^Read_92/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0l
Identity_184IdentityRead_92/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2c
Identity_185IdentityIdentity_184:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_93/DisableCopyOnReadDisableCopyOnRead/read_93_disablecopyonread_v_conv_prep1_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_93/ReadVariableOpReadVariableOp/read_93_disablecopyonread_v_conv_prep1_3x3_bias^Read_93/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0l
Identity_186IdentityRead_93/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2c
Identity_187IdentityIdentity_186:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_94/DisableCopyOnReadDisableCopyOnRead1read_94_disablecopyonread_m_conv_prep1_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_94/ReadVariableOpReadVariableOp1read_94_disablecopyonread_m_conv_prep1_4x4_kernel^Read_94/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A
*
dtype0x
Identity_188IdentityRead_94/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A
o
Identity_189IdentityIdentity_188:output:0"/device:CPU:0*
T0*&
_output_shapes
:A
�
Read_95/DisableCopyOnReadDisableCopyOnRead1read_95_disablecopyonread_v_conv_prep1_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_95/ReadVariableOpReadVariableOp1read_95_disablecopyonread_v_conv_prep1_4x4_kernel^Read_95/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A
*
dtype0x
Identity_190IdentityRead_95/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A
o
Identity_191IdentityIdentity_190:output:0"/device:CPU:0*
T0*&
_output_shapes
:A
�
Read_96/DisableCopyOnReadDisableCopyOnRead/read_96_disablecopyonread_m_conv_prep1_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_96/ReadVariableOpReadVariableOp/read_96_disablecopyonread_m_conv_prep1_4x4_bias^Read_96/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0l
Identity_192IdentityRead_96/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_193IdentityIdentity_192:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_97/DisableCopyOnReadDisableCopyOnRead/read_97_disablecopyonread_v_conv_prep1_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_97/ReadVariableOpReadVariableOp/read_97_disablecopyonread_v_conv_prep1_4x4_bias^Read_97/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0l
Identity_194IdentityRead_97/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_195IdentityIdentity_194:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_98/DisableCopyOnReadDisableCopyOnRead1read_98_disablecopyonread_m_conv_prep1_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_98/ReadVariableOpReadVariableOp1read_98_disablecopyonread_m_conv_prep1_5x5_kernel^Read_98/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0x
Identity_196IdentityRead_98/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Ao
Identity_197IdentityIdentity_196:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_99/DisableCopyOnReadDisableCopyOnRead1read_99_disablecopyonread_v_conv_prep1_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_99/ReadVariableOpReadVariableOp1read_99_disablecopyonread_v_conv_prep1_5x5_kernel^Read_99/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0x
Identity_198IdentityRead_99/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Ao
Identity_199IdentityIdentity_198:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_100/DisableCopyOnReadDisableCopyOnRead0read_100_disablecopyonread_m_conv_prep1_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_100/ReadVariableOpReadVariableOp0read_100_disablecopyonread_m_conv_prep1_5x5_bias^Read_100/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_200IdentityRead_100/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_201IdentityIdentity_200:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_101/DisableCopyOnReadDisableCopyOnRead0read_101_disablecopyonread_v_conv_prep1_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_101/ReadVariableOpReadVariableOp0read_101_disablecopyonread_v_conv_prep1_5x5_bias^Read_101/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_202IdentityRead_101/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_203IdentityIdentity_202:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_102/DisableCopyOnReadDisableCopyOnRead1read_102_disablecopyonread_m_conv_hid0_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_102/ReadVariableOpReadVariableOp1read_102_disablecopyonread_m_conv_hid0_3x3_kernel^Read_102/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:D2*
dtype0y
Identity_204IdentityRead_102/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:D2o
Identity_205IdentityIdentity_204:output:0"/device:CPU:0*
T0*&
_output_shapes
:D2�
Read_103/DisableCopyOnReadDisableCopyOnRead1read_103_disablecopyonread_v_conv_hid0_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_103/ReadVariableOpReadVariableOp1read_103_disablecopyonread_v_conv_hid0_3x3_kernel^Read_103/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:D2*
dtype0y
Identity_206IdentityRead_103/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:D2o
Identity_207IdentityIdentity_206:output:0"/device:CPU:0*
T0*&
_output_shapes
:D2�
Read_104/DisableCopyOnReadDisableCopyOnRead/read_104_disablecopyonread_m_conv_hid0_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_104/ReadVariableOpReadVariableOp/read_104_disablecopyonread_m_conv_hid0_3x3_bias^Read_104/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0m
Identity_208IdentityRead_104/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2c
Identity_209IdentityIdentity_208:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_105/DisableCopyOnReadDisableCopyOnRead/read_105_disablecopyonread_v_conv_hid0_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_105/ReadVariableOpReadVariableOp/read_105_disablecopyonread_v_conv_hid0_3x3_bias^Read_105/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0m
Identity_210IdentityRead_105/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2c
Identity_211IdentityIdentity_210:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_106/DisableCopyOnReadDisableCopyOnRead1read_106_disablecopyonread_m_conv_hid0_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_106/ReadVariableOpReadVariableOp1read_106_disablecopyonread_m_conv_hid0_4x4_kernel^Read_106/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:D
*
dtype0y
Identity_212IdentityRead_106/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:D
o
Identity_213IdentityIdentity_212:output:0"/device:CPU:0*
T0*&
_output_shapes
:D
�
Read_107/DisableCopyOnReadDisableCopyOnRead1read_107_disablecopyonread_v_conv_hid0_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_107/ReadVariableOpReadVariableOp1read_107_disablecopyonread_v_conv_hid0_4x4_kernel^Read_107/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:D
*
dtype0y
Identity_214IdentityRead_107/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:D
o
Identity_215IdentityIdentity_214:output:0"/device:CPU:0*
T0*&
_output_shapes
:D
�
Read_108/DisableCopyOnReadDisableCopyOnRead/read_108_disablecopyonread_m_conv_hid0_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_108/ReadVariableOpReadVariableOp/read_108_disablecopyonread_m_conv_hid0_4x4_bias^Read_108/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0m
Identity_216IdentityRead_108/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_217IdentityIdentity_216:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_109/DisableCopyOnReadDisableCopyOnRead/read_109_disablecopyonread_v_conv_hid0_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_109/ReadVariableOpReadVariableOp/read_109_disablecopyonread_v_conv_hid0_4x4_bias^Read_109/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0m
Identity_218IdentityRead_109/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_219IdentityIdentity_218:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_110/DisableCopyOnReadDisableCopyOnRead1read_110_disablecopyonread_m_conv_hid0_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_110/ReadVariableOpReadVariableOp1read_110_disablecopyonread_m_conv_hid0_5x5_kernel^Read_110/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:D*
dtype0y
Identity_220IdentityRead_110/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Do
Identity_221IdentityIdentity_220:output:0"/device:CPU:0*
T0*&
_output_shapes
:D�
Read_111/DisableCopyOnReadDisableCopyOnRead1read_111_disablecopyonread_v_conv_hid0_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_111/ReadVariableOpReadVariableOp1read_111_disablecopyonread_v_conv_hid0_5x5_kernel^Read_111/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:D*
dtype0y
Identity_222IdentityRead_111/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Do
Identity_223IdentityIdentity_222:output:0"/device:CPU:0*
T0*&
_output_shapes
:D�
Read_112/DisableCopyOnReadDisableCopyOnRead/read_112_disablecopyonread_m_conv_hid0_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_112/ReadVariableOpReadVariableOp/read_112_disablecopyonread_m_conv_hid0_5x5_bias^Read_112/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_224IdentityRead_112/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_225IdentityIdentity_224:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_113/DisableCopyOnReadDisableCopyOnRead/read_113_disablecopyonread_v_conv_hid0_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_113/ReadVariableOpReadVariableOp/read_113_disablecopyonread_v_conv_hid0_5x5_bias^Read_113/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_226IdentityRead_113/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_227IdentityIdentity_226:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_114/DisableCopyOnReadDisableCopyOnRead1read_114_disablecopyonread_m_conv_hid1_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_114/ReadVariableOpReadVariableOp1read_114_disablecopyonread_m_conv_hid1_3x3_kernel^Read_114/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A2*
dtype0y
Identity_228IdentityRead_114/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A2o
Identity_229IdentityIdentity_228:output:0"/device:CPU:0*
T0*&
_output_shapes
:A2�
Read_115/DisableCopyOnReadDisableCopyOnRead1read_115_disablecopyonread_v_conv_hid1_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_115/ReadVariableOpReadVariableOp1read_115_disablecopyonread_v_conv_hid1_3x3_kernel^Read_115/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A2*
dtype0y
Identity_230IdentityRead_115/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A2o
Identity_231IdentityIdentity_230:output:0"/device:CPU:0*
T0*&
_output_shapes
:A2�
Read_116/DisableCopyOnReadDisableCopyOnRead/read_116_disablecopyonread_m_conv_hid1_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_116/ReadVariableOpReadVariableOp/read_116_disablecopyonread_m_conv_hid1_3x3_bias^Read_116/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0m
Identity_232IdentityRead_116/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2c
Identity_233IdentityIdentity_232:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_117/DisableCopyOnReadDisableCopyOnRead/read_117_disablecopyonread_v_conv_hid1_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_117/ReadVariableOpReadVariableOp/read_117_disablecopyonread_v_conv_hid1_3x3_bias^Read_117/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0m
Identity_234IdentityRead_117/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2c
Identity_235IdentityIdentity_234:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_118/DisableCopyOnReadDisableCopyOnRead1read_118_disablecopyonread_m_conv_hid1_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_118/ReadVariableOpReadVariableOp1read_118_disablecopyonread_m_conv_hid1_4x4_kernel^Read_118/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A
*
dtype0y
Identity_236IdentityRead_118/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A
o
Identity_237IdentityIdentity_236:output:0"/device:CPU:0*
T0*&
_output_shapes
:A
�
Read_119/DisableCopyOnReadDisableCopyOnRead1read_119_disablecopyonread_v_conv_hid1_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_119/ReadVariableOpReadVariableOp1read_119_disablecopyonread_v_conv_hid1_4x4_kernel^Read_119/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A
*
dtype0y
Identity_238IdentityRead_119/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A
o
Identity_239IdentityIdentity_238:output:0"/device:CPU:0*
T0*&
_output_shapes
:A
�
Read_120/DisableCopyOnReadDisableCopyOnRead/read_120_disablecopyonread_m_conv_hid1_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_120/ReadVariableOpReadVariableOp/read_120_disablecopyonread_m_conv_hid1_4x4_bias^Read_120/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0m
Identity_240IdentityRead_120/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_241IdentityIdentity_240:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_121/DisableCopyOnReadDisableCopyOnRead/read_121_disablecopyonread_v_conv_hid1_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_121/ReadVariableOpReadVariableOp/read_121_disablecopyonread_v_conv_hid1_4x4_bias^Read_121/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0m
Identity_242IdentityRead_121/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_243IdentityIdentity_242:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_122/DisableCopyOnReadDisableCopyOnRead1read_122_disablecopyonread_m_conv_hid1_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_122/ReadVariableOpReadVariableOp1read_122_disablecopyonread_m_conv_hid1_5x5_kernel^Read_122/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0y
Identity_244IdentityRead_122/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Ao
Identity_245IdentityIdentity_244:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_123/DisableCopyOnReadDisableCopyOnRead1read_123_disablecopyonread_v_conv_hid1_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_123/ReadVariableOpReadVariableOp1read_123_disablecopyonread_v_conv_hid1_5x5_kernel^Read_123/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0y
Identity_246IdentityRead_123/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Ao
Identity_247IdentityIdentity_246:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_124/DisableCopyOnReadDisableCopyOnRead/read_124_disablecopyonread_m_conv_hid1_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_124/ReadVariableOpReadVariableOp/read_124_disablecopyonread_m_conv_hid1_5x5_bias^Read_124/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_248IdentityRead_124/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_249IdentityIdentity_248:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_125/DisableCopyOnReadDisableCopyOnRead/read_125_disablecopyonread_v_conv_hid1_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_125/ReadVariableOpReadVariableOp/read_125_disablecopyonread_v_conv_hid1_5x5_bias^Read_125/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_250IdentityRead_125/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_251IdentityIdentity_250:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_126/DisableCopyOnReadDisableCopyOnRead1read_126_disablecopyonread_m_conv_hid2_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_126/ReadVariableOpReadVariableOp1read_126_disablecopyonread_m_conv_hid2_3x3_kernel^Read_126/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A2*
dtype0y
Identity_252IdentityRead_126/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A2o
Identity_253IdentityIdentity_252:output:0"/device:CPU:0*
T0*&
_output_shapes
:A2�
Read_127/DisableCopyOnReadDisableCopyOnRead1read_127_disablecopyonread_v_conv_hid2_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_127/ReadVariableOpReadVariableOp1read_127_disablecopyonread_v_conv_hid2_3x3_kernel^Read_127/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A2*
dtype0y
Identity_254IdentityRead_127/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A2o
Identity_255IdentityIdentity_254:output:0"/device:CPU:0*
T0*&
_output_shapes
:A2�
Read_128/DisableCopyOnReadDisableCopyOnRead/read_128_disablecopyonread_m_conv_hid2_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_128/ReadVariableOpReadVariableOp/read_128_disablecopyonread_m_conv_hid2_3x3_bias^Read_128/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0m
Identity_256IdentityRead_128/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2c
Identity_257IdentityIdentity_256:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_129/DisableCopyOnReadDisableCopyOnRead/read_129_disablecopyonread_v_conv_hid2_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_129/ReadVariableOpReadVariableOp/read_129_disablecopyonread_v_conv_hid2_3x3_bias^Read_129/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0m
Identity_258IdentityRead_129/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2c
Identity_259IdentityIdentity_258:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_130/DisableCopyOnReadDisableCopyOnRead1read_130_disablecopyonread_m_conv_hid2_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_130/ReadVariableOpReadVariableOp1read_130_disablecopyonread_m_conv_hid2_4x4_kernel^Read_130/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A
*
dtype0y
Identity_260IdentityRead_130/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A
o
Identity_261IdentityIdentity_260:output:0"/device:CPU:0*
T0*&
_output_shapes
:A
�
Read_131/DisableCopyOnReadDisableCopyOnRead1read_131_disablecopyonread_v_conv_hid2_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_131/ReadVariableOpReadVariableOp1read_131_disablecopyonread_v_conv_hid2_4x4_kernel^Read_131/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A
*
dtype0y
Identity_262IdentityRead_131/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A
o
Identity_263IdentityIdentity_262:output:0"/device:CPU:0*
T0*&
_output_shapes
:A
�
Read_132/DisableCopyOnReadDisableCopyOnRead/read_132_disablecopyonread_m_conv_hid2_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_132/ReadVariableOpReadVariableOp/read_132_disablecopyonread_m_conv_hid2_4x4_bias^Read_132/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0m
Identity_264IdentityRead_132/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_265IdentityIdentity_264:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_133/DisableCopyOnReadDisableCopyOnRead/read_133_disablecopyonread_v_conv_hid2_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_133/ReadVariableOpReadVariableOp/read_133_disablecopyonread_v_conv_hid2_4x4_bias^Read_133/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0m
Identity_266IdentityRead_133/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_267IdentityIdentity_266:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_134/DisableCopyOnReadDisableCopyOnRead1read_134_disablecopyonread_m_conv_hid2_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_134/ReadVariableOpReadVariableOp1read_134_disablecopyonread_m_conv_hid2_5x5_kernel^Read_134/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0y
Identity_268IdentityRead_134/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Ao
Identity_269IdentityIdentity_268:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_135/DisableCopyOnReadDisableCopyOnRead1read_135_disablecopyonread_v_conv_hid2_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_135/ReadVariableOpReadVariableOp1read_135_disablecopyonread_v_conv_hid2_5x5_kernel^Read_135/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0y
Identity_270IdentityRead_135/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Ao
Identity_271IdentityIdentity_270:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_136/DisableCopyOnReadDisableCopyOnRead/read_136_disablecopyonread_m_conv_hid2_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_136/ReadVariableOpReadVariableOp/read_136_disablecopyonread_m_conv_hid2_5x5_bias^Read_136/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_272IdentityRead_136/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_273IdentityIdentity_272:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_137/DisableCopyOnReadDisableCopyOnRead/read_137_disablecopyonread_v_conv_hid2_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_137/ReadVariableOpReadVariableOp/read_137_disablecopyonread_v_conv_hid2_5x5_bias^Read_137/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_274IdentityRead_137/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_275IdentityIdentity_274:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_138/DisableCopyOnReadDisableCopyOnRead1read_138_disablecopyonread_m_conv_hid3_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_138/ReadVariableOpReadVariableOp1read_138_disablecopyonread_m_conv_hid3_3x3_kernel^Read_138/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A2*
dtype0y
Identity_276IdentityRead_138/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A2o
Identity_277IdentityIdentity_276:output:0"/device:CPU:0*
T0*&
_output_shapes
:A2�
Read_139/DisableCopyOnReadDisableCopyOnRead1read_139_disablecopyonread_v_conv_hid3_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_139/ReadVariableOpReadVariableOp1read_139_disablecopyonread_v_conv_hid3_3x3_kernel^Read_139/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A2*
dtype0y
Identity_278IdentityRead_139/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A2o
Identity_279IdentityIdentity_278:output:0"/device:CPU:0*
T0*&
_output_shapes
:A2�
Read_140/DisableCopyOnReadDisableCopyOnRead/read_140_disablecopyonread_m_conv_hid3_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_140/ReadVariableOpReadVariableOp/read_140_disablecopyonread_m_conv_hid3_3x3_bias^Read_140/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0m
Identity_280IdentityRead_140/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2c
Identity_281IdentityIdentity_280:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_141/DisableCopyOnReadDisableCopyOnRead/read_141_disablecopyonread_v_conv_hid3_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_141/ReadVariableOpReadVariableOp/read_141_disablecopyonread_v_conv_hid3_3x3_bias^Read_141/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0m
Identity_282IdentityRead_141/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2c
Identity_283IdentityIdentity_282:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_142/DisableCopyOnReadDisableCopyOnRead1read_142_disablecopyonread_m_conv_hid3_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_142/ReadVariableOpReadVariableOp1read_142_disablecopyonread_m_conv_hid3_4x4_kernel^Read_142/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A
*
dtype0y
Identity_284IdentityRead_142/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A
o
Identity_285IdentityIdentity_284:output:0"/device:CPU:0*
T0*&
_output_shapes
:A
�
Read_143/DisableCopyOnReadDisableCopyOnRead1read_143_disablecopyonread_v_conv_hid3_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_143/ReadVariableOpReadVariableOp1read_143_disablecopyonread_v_conv_hid3_4x4_kernel^Read_143/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A
*
dtype0y
Identity_286IdentityRead_143/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A
o
Identity_287IdentityIdentity_286:output:0"/device:CPU:0*
T0*&
_output_shapes
:A
�
Read_144/DisableCopyOnReadDisableCopyOnRead/read_144_disablecopyonread_m_conv_hid3_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_144/ReadVariableOpReadVariableOp/read_144_disablecopyonread_m_conv_hid3_4x4_bias^Read_144/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0m
Identity_288IdentityRead_144/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_289IdentityIdentity_288:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_145/DisableCopyOnReadDisableCopyOnRead/read_145_disablecopyonread_v_conv_hid3_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_145/ReadVariableOpReadVariableOp/read_145_disablecopyonread_v_conv_hid3_4x4_bias^Read_145/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0m
Identity_290IdentityRead_145/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_291IdentityIdentity_290:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_146/DisableCopyOnReadDisableCopyOnRead1read_146_disablecopyonread_m_conv_hid3_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_146/ReadVariableOpReadVariableOp1read_146_disablecopyonread_m_conv_hid3_5x5_kernel^Read_146/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0y
Identity_292IdentityRead_146/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Ao
Identity_293IdentityIdentity_292:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_147/DisableCopyOnReadDisableCopyOnRead1read_147_disablecopyonread_v_conv_hid3_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_147/ReadVariableOpReadVariableOp1read_147_disablecopyonread_v_conv_hid3_5x5_kernel^Read_147/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0y
Identity_294IdentityRead_147/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Ao
Identity_295IdentityIdentity_294:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_148/DisableCopyOnReadDisableCopyOnRead/read_148_disablecopyonread_m_conv_hid3_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_148/ReadVariableOpReadVariableOp/read_148_disablecopyonread_m_conv_hid3_5x5_bias^Read_148/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_296IdentityRead_148/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_297IdentityIdentity_296:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_149/DisableCopyOnReadDisableCopyOnRead/read_149_disablecopyonread_v_conv_hid3_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_149/ReadVariableOpReadVariableOp/read_149_disablecopyonread_v_conv_hid3_5x5_bias^Read_149/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_298IdentityRead_149/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_299IdentityIdentity_298:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_150/DisableCopyOnReadDisableCopyOnRead1read_150_disablecopyonread_m_conv_hid4_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_150/ReadVariableOpReadVariableOp1read_150_disablecopyonread_m_conv_hid4_3x3_kernel^Read_150/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A2*
dtype0y
Identity_300IdentityRead_150/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A2o
Identity_301IdentityIdentity_300:output:0"/device:CPU:0*
T0*&
_output_shapes
:A2�
Read_151/DisableCopyOnReadDisableCopyOnRead1read_151_disablecopyonread_v_conv_hid4_3x3_kernel"/device:CPU:0*
_output_shapes
 �
Read_151/ReadVariableOpReadVariableOp1read_151_disablecopyonread_v_conv_hid4_3x3_kernel^Read_151/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A2*
dtype0y
Identity_302IdentityRead_151/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A2o
Identity_303IdentityIdentity_302:output:0"/device:CPU:0*
T0*&
_output_shapes
:A2�
Read_152/DisableCopyOnReadDisableCopyOnRead/read_152_disablecopyonread_m_conv_hid4_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_152/ReadVariableOpReadVariableOp/read_152_disablecopyonread_m_conv_hid4_3x3_bias^Read_152/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0m
Identity_304IdentityRead_152/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2c
Identity_305IdentityIdentity_304:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_153/DisableCopyOnReadDisableCopyOnRead/read_153_disablecopyonread_v_conv_hid4_3x3_bias"/device:CPU:0*
_output_shapes
 �
Read_153/ReadVariableOpReadVariableOp/read_153_disablecopyonread_v_conv_hid4_3x3_bias^Read_153/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0m
Identity_306IdentityRead_153/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2c
Identity_307IdentityIdentity_306:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_154/DisableCopyOnReadDisableCopyOnRead1read_154_disablecopyonread_m_conv_hid4_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_154/ReadVariableOpReadVariableOp1read_154_disablecopyonread_m_conv_hid4_4x4_kernel^Read_154/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A
*
dtype0y
Identity_308IdentityRead_154/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A
o
Identity_309IdentityIdentity_308:output:0"/device:CPU:0*
T0*&
_output_shapes
:A
�
Read_155/DisableCopyOnReadDisableCopyOnRead1read_155_disablecopyonread_v_conv_hid4_4x4_kernel"/device:CPU:0*
_output_shapes
 �
Read_155/ReadVariableOpReadVariableOp1read_155_disablecopyonread_v_conv_hid4_4x4_kernel^Read_155/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A
*
dtype0y
Identity_310IdentityRead_155/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:A
o
Identity_311IdentityIdentity_310:output:0"/device:CPU:0*
T0*&
_output_shapes
:A
�
Read_156/DisableCopyOnReadDisableCopyOnRead/read_156_disablecopyonread_m_conv_hid4_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_156/ReadVariableOpReadVariableOp/read_156_disablecopyonread_m_conv_hid4_4x4_bias^Read_156/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0m
Identity_312IdentityRead_156/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_313IdentityIdentity_312:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_157/DisableCopyOnReadDisableCopyOnRead/read_157_disablecopyonread_v_conv_hid4_4x4_bias"/device:CPU:0*
_output_shapes
 �
Read_157/ReadVariableOpReadVariableOp/read_157_disablecopyonread_v_conv_hid4_4x4_bias^Read_157/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0m
Identity_314IdentityRead_157/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_315IdentityIdentity_314:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_158/DisableCopyOnReadDisableCopyOnRead1read_158_disablecopyonread_m_conv_hid5_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_158/ReadVariableOpReadVariableOp1read_158_disablecopyonread_m_conv_hid5_5x5_kernel^Read_158/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0y
Identity_316IdentityRead_158/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Ao
Identity_317IdentityIdentity_316:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_159/DisableCopyOnReadDisableCopyOnRead1read_159_disablecopyonread_v_conv_hid5_5x5_kernel"/device:CPU:0*
_output_shapes
 �
Read_159/ReadVariableOpReadVariableOp1read_159_disablecopyonread_v_conv_hid5_5x5_kernel^Read_159/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0y
Identity_318IdentityRead_159/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Ao
Identity_319IdentityIdentity_318:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_160/DisableCopyOnReadDisableCopyOnRead/read_160_disablecopyonread_m_conv_hid5_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_160/ReadVariableOpReadVariableOp/read_160_disablecopyonread_m_conv_hid5_5x5_bias^Read_160/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_320IdentityRead_160/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_321IdentityIdentity_320:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_161/DisableCopyOnReadDisableCopyOnRead/read_161_disablecopyonread_v_conv_hid5_5x5_bias"/device:CPU:0*
_output_shapes
 �
Read_161/ReadVariableOpReadVariableOp/read_161_disablecopyonread_v_conv_hid5_5x5_bias^Read_161/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_322IdentityRead_161/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_323IdentityIdentity_322:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_162/DisableCopyOnReadDisableCopyOnRead,read_162_disablecopyonread_m_output_c_kernel"/device:CPU:0*
_output_shapes
 �
Read_162/ReadVariableOpReadVariableOp,read_162_disablecopyonread_m_output_c_kernel^Read_162/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0y
Identity_324IdentityRead_162/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Ao
Identity_325IdentityIdentity_324:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_163/DisableCopyOnReadDisableCopyOnRead,read_163_disablecopyonread_v_output_c_kernel"/device:CPU:0*
_output_shapes
 �
Read_163/ReadVariableOpReadVariableOp,read_163_disablecopyonread_v_output_c_kernel^Read_163/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0y
Identity_326IdentityRead_163/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Ao
Identity_327IdentityIdentity_326:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_164/DisableCopyOnReadDisableCopyOnRead*read_164_disablecopyonread_m_output_c_bias"/device:CPU:0*
_output_shapes
 �
Read_164/ReadVariableOpReadVariableOp*read_164_disablecopyonread_m_output_c_bias^Read_164/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_328IdentityRead_164/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_329IdentityIdentity_328:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_165/DisableCopyOnReadDisableCopyOnRead*read_165_disablecopyonread_v_output_c_bias"/device:CPU:0*
_output_shapes
 �
Read_165/ReadVariableOpReadVariableOp*read_165_disablecopyonread_v_output_c_bias^Read_165/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_330IdentityRead_165/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_331IdentityIdentity_330:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_166/DisableCopyOnReadDisableCopyOnRead read_166_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_166/ReadVariableOpReadVariableOp read_166_disablecopyonread_total^Read_166/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_332IdentityRead_166/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_333IdentityIdentity_332:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_167/DisableCopyOnReadDisableCopyOnRead read_167_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_167/ReadVariableOpReadVariableOp read_167_disablecopyonread_count^Read_167/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_334IdentityRead_167/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_335IdentityIdentity_334:output:0"/device:CPU:0*
T0*
_output_shapes
: �>
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�>
value�=B�=�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB'variables/56/.ATTRIBUTES/VARIABLE_VALUEB'variables/57/.ATTRIBUTES/VARIABLE_VALUEB'variables/58/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB'variables/60/.ATTRIBUTES/VARIABLE_VALUEB'variables/61/.ATTRIBUTES/VARIABLE_VALUEB'variables/62/.ATTRIBUTES/VARIABLE_VALUEB'variables/63/.ATTRIBUTES/VARIABLE_VALUEB'variables/64/.ATTRIBUTES/VARIABLE_VALUEB'variables/65/.ATTRIBUTES/VARIABLE_VALUEB'variables/66/.ATTRIBUTES/VARIABLE_VALUEB'variables/67/.ATTRIBUTES/VARIABLE_VALUEB'variables/68/.ATTRIBUTES/VARIABLE_VALUEB'variables/69/.ATTRIBUTES/VARIABLE_VALUEB'variables/70/.ATTRIBUTES/VARIABLE_VALUEB'variables/71/.ATTRIBUTES/VARIABLE_VALUEB'variables/72/.ATTRIBUTES/VARIABLE_VALUEB'variables/73/.ATTRIBUTES/VARIABLE_VALUEB'variables/74/.ATTRIBUTES/VARIABLE_VALUEB'variables/75/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/73/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/74/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/75/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/76/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/77/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/78/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/79/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/80/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/81/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/82/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/83/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/84/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/85/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/86/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/87/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/88/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B � 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0Identity_161:output:0Identity_163:output:0Identity_165:output:0Identity_167:output:0Identity_169:output:0Identity_171:output:0Identity_173:output:0Identity_175:output:0Identity_177:output:0Identity_179:output:0Identity_181:output:0Identity_183:output:0Identity_185:output:0Identity_187:output:0Identity_189:output:0Identity_191:output:0Identity_193:output:0Identity_195:output:0Identity_197:output:0Identity_199:output:0Identity_201:output:0Identity_203:output:0Identity_205:output:0Identity_207:output:0Identity_209:output:0Identity_211:output:0Identity_213:output:0Identity_215:output:0Identity_217:output:0Identity_219:output:0Identity_221:output:0Identity_223:output:0Identity_225:output:0Identity_227:output:0Identity_229:output:0Identity_231:output:0Identity_233:output:0Identity_235:output:0Identity_237:output:0Identity_239:output:0Identity_241:output:0Identity_243:output:0Identity_245:output:0Identity_247:output:0Identity_249:output:0Identity_251:output:0Identity_253:output:0Identity_255:output:0Identity_257:output:0Identity_259:output:0Identity_261:output:0Identity_263:output:0Identity_265:output:0Identity_267:output:0Identity_269:output:0Identity_271:output:0Identity_273:output:0Identity_275:output:0Identity_277:output:0Identity_279:output:0Identity_281:output:0Identity_283:output:0Identity_285:output:0Identity_287:output:0Identity_289:output:0Identity_291:output:0Identity_293:output:0Identity_295:output:0Identity_297:output:0Identity_299:output:0Identity_301:output:0Identity_303:output:0Identity_305:output:0Identity_307:output:0Identity_309:output:0Identity_311:output:0Identity_313:output:0Identity_315:output:0Identity_317:output:0Identity_319:output:0Identity_321:output:0Identity_323:output:0Identity_325:output:0Identity_327:output:0Identity_329:output:0Identity_331:output:0Identity_333:output:0Identity_335:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *�
dtypes�
�2�	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_336Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_337IdentityIdentity_336:output:0^NoOp*
T0*
_output_shapes
: �F
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_100/DisableCopyOnRead^Read_100/ReadVariableOp^Read_101/DisableCopyOnRead^Read_101/ReadVariableOp^Read_102/DisableCopyOnRead^Read_102/ReadVariableOp^Read_103/DisableCopyOnRead^Read_103/ReadVariableOp^Read_104/DisableCopyOnRead^Read_104/ReadVariableOp^Read_105/DisableCopyOnRead^Read_105/ReadVariableOp^Read_106/DisableCopyOnRead^Read_106/ReadVariableOp^Read_107/DisableCopyOnRead^Read_107/ReadVariableOp^Read_108/DisableCopyOnRead^Read_108/ReadVariableOp^Read_109/DisableCopyOnRead^Read_109/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_110/DisableCopyOnRead^Read_110/ReadVariableOp^Read_111/DisableCopyOnRead^Read_111/ReadVariableOp^Read_112/DisableCopyOnRead^Read_112/ReadVariableOp^Read_113/DisableCopyOnRead^Read_113/ReadVariableOp^Read_114/DisableCopyOnRead^Read_114/ReadVariableOp^Read_115/DisableCopyOnRead^Read_115/ReadVariableOp^Read_116/DisableCopyOnRead^Read_116/ReadVariableOp^Read_117/DisableCopyOnRead^Read_117/ReadVariableOp^Read_118/DisableCopyOnRead^Read_118/ReadVariableOp^Read_119/DisableCopyOnRead^Read_119/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_120/DisableCopyOnRead^Read_120/ReadVariableOp^Read_121/DisableCopyOnRead^Read_121/ReadVariableOp^Read_122/DisableCopyOnRead^Read_122/ReadVariableOp^Read_123/DisableCopyOnRead^Read_123/ReadVariableOp^Read_124/DisableCopyOnRead^Read_124/ReadVariableOp^Read_125/DisableCopyOnRead^Read_125/ReadVariableOp^Read_126/DisableCopyOnRead^Read_126/ReadVariableOp^Read_127/DisableCopyOnRead^Read_127/ReadVariableOp^Read_128/DisableCopyOnRead^Read_128/ReadVariableOp^Read_129/DisableCopyOnRead^Read_129/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_130/DisableCopyOnRead^Read_130/ReadVariableOp^Read_131/DisableCopyOnRead^Read_131/ReadVariableOp^Read_132/DisableCopyOnRead^Read_132/ReadVariableOp^Read_133/DisableCopyOnRead^Read_133/ReadVariableOp^Read_134/DisableCopyOnRead^Read_134/ReadVariableOp^Read_135/DisableCopyOnRead^Read_135/ReadVariableOp^Read_136/DisableCopyOnRead^Read_136/ReadVariableOp^Read_137/DisableCopyOnRead^Read_137/ReadVariableOp^Read_138/DisableCopyOnRead^Read_138/ReadVariableOp^Read_139/DisableCopyOnRead^Read_139/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_140/DisableCopyOnRead^Read_140/ReadVariableOp^Read_141/DisableCopyOnRead^Read_141/ReadVariableOp^Read_142/DisableCopyOnRead^Read_142/ReadVariableOp^Read_143/DisableCopyOnRead^Read_143/ReadVariableOp^Read_144/DisableCopyOnRead^Read_144/ReadVariableOp^Read_145/DisableCopyOnRead^Read_145/ReadVariableOp^Read_146/DisableCopyOnRead^Read_146/ReadVariableOp^Read_147/DisableCopyOnRead^Read_147/ReadVariableOp^Read_148/DisableCopyOnRead^Read_148/ReadVariableOp^Read_149/DisableCopyOnRead^Read_149/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_150/DisableCopyOnRead^Read_150/ReadVariableOp^Read_151/DisableCopyOnRead^Read_151/ReadVariableOp^Read_152/DisableCopyOnRead^Read_152/ReadVariableOp^Read_153/DisableCopyOnRead^Read_153/ReadVariableOp^Read_154/DisableCopyOnRead^Read_154/ReadVariableOp^Read_155/DisableCopyOnRead^Read_155/ReadVariableOp^Read_156/DisableCopyOnRead^Read_156/ReadVariableOp^Read_157/DisableCopyOnRead^Read_157/ReadVariableOp^Read_158/DisableCopyOnRead^Read_158/ReadVariableOp^Read_159/DisableCopyOnRead^Read_159/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_160/DisableCopyOnRead^Read_160/ReadVariableOp^Read_161/DisableCopyOnRead^Read_161/ReadVariableOp^Read_162/DisableCopyOnRead^Read_162/ReadVariableOp^Read_163/DisableCopyOnRead^Read_163/ReadVariableOp^Read_164/DisableCopyOnRead^Read_164/ReadVariableOp^Read_165/DisableCopyOnRead^Read_165/ReadVariableOp^Read_166/DisableCopyOnRead^Read_166/ReadVariableOp^Read_167/DisableCopyOnRead^Read_167/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_80/DisableCopyOnRead^Read_80/ReadVariableOp^Read_81/DisableCopyOnRead^Read_81/ReadVariableOp^Read_82/DisableCopyOnRead^Read_82/ReadVariableOp^Read_83/DisableCopyOnRead^Read_83/ReadVariableOp^Read_84/DisableCopyOnRead^Read_84/ReadVariableOp^Read_85/DisableCopyOnRead^Read_85/ReadVariableOp^Read_86/DisableCopyOnRead^Read_86/ReadVariableOp^Read_87/DisableCopyOnRead^Read_87/ReadVariableOp^Read_88/DisableCopyOnRead^Read_88/ReadVariableOp^Read_89/DisableCopyOnRead^Read_89/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp^Read_90/DisableCopyOnRead^Read_90/ReadVariableOp^Read_91/DisableCopyOnRead^Read_91/ReadVariableOp^Read_92/DisableCopyOnRead^Read_92/ReadVariableOp^Read_93/DisableCopyOnRead^Read_93/ReadVariableOp^Read_94/DisableCopyOnRead^Read_94/ReadVariableOp^Read_95/DisableCopyOnRead^Read_95/ReadVariableOp^Read_96/DisableCopyOnRead^Read_96/ReadVariableOp^Read_97/DisableCopyOnRead^Read_97/ReadVariableOp^Read_98/DisableCopyOnRead^Read_98/ReadVariableOp^Read_99/DisableCopyOnRead^Read_99/ReadVariableOp*
_output_shapes
 "%
identity_337Identity_337:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp28
Read_100/DisableCopyOnReadRead_100/DisableCopyOnRead22
Read_100/ReadVariableOpRead_100/ReadVariableOp28
Read_101/DisableCopyOnReadRead_101/DisableCopyOnRead22
Read_101/ReadVariableOpRead_101/ReadVariableOp28
Read_102/DisableCopyOnReadRead_102/DisableCopyOnRead22
Read_102/ReadVariableOpRead_102/ReadVariableOp28
Read_103/DisableCopyOnReadRead_103/DisableCopyOnRead22
Read_103/ReadVariableOpRead_103/ReadVariableOp28
Read_104/DisableCopyOnReadRead_104/DisableCopyOnRead22
Read_104/ReadVariableOpRead_104/ReadVariableOp28
Read_105/DisableCopyOnReadRead_105/DisableCopyOnRead22
Read_105/ReadVariableOpRead_105/ReadVariableOp28
Read_106/DisableCopyOnReadRead_106/DisableCopyOnRead22
Read_106/ReadVariableOpRead_106/ReadVariableOp28
Read_107/DisableCopyOnReadRead_107/DisableCopyOnRead22
Read_107/ReadVariableOpRead_107/ReadVariableOp28
Read_108/DisableCopyOnReadRead_108/DisableCopyOnRead22
Read_108/ReadVariableOpRead_108/ReadVariableOp28
Read_109/DisableCopyOnReadRead_109/DisableCopyOnRead22
Read_109/ReadVariableOpRead_109/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp28
Read_110/DisableCopyOnReadRead_110/DisableCopyOnRead22
Read_110/ReadVariableOpRead_110/ReadVariableOp28
Read_111/DisableCopyOnReadRead_111/DisableCopyOnRead22
Read_111/ReadVariableOpRead_111/ReadVariableOp28
Read_112/DisableCopyOnReadRead_112/DisableCopyOnRead22
Read_112/ReadVariableOpRead_112/ReadVariableOp28
Read_113/DisableCopyOnReadRead_113/DisableCopyOnRead22
Read_113/ReadVariableOpRead_113/ReadVariableOp28
Read_114/DisableCopyOnReadRead_114/DisableCopyOnRead22
Read_114/ReadVariableOpRead_114/ReadVariableOp28
Read_115/DisableCopyOnReadRead_115/DisableCopyOnRead22
Read_115/ReadVariableOpRead_115/ReadVariableOp28
Read_116/DisableCopyOnReadRead_116/DisableCopyOnRead22
Read_116/ReadVariableOpRead_116/ReadVariableOp28
Read_117/DisableCopyOnReadRead_117/DisableCopyOnRead22
Read_117/ReadVariableOpRead_117/ReadVariableOp28
Read_118/DisableCopyOnReadRead_118/DisableCopyOnRead22
Read_118/ReadVariableOpRead_118/ReadVariableOp28
Read_119/DisableCopyOnReadRead_119/DisableCopyOnRead22
Read_119/ReadVariableOpRead_119/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp28
Read_120/DisableCopyOnReadRead_120/DisableCopyOnRead22
Read_120/ReadVariableOpRead_120/ReadVariableOp28
Read_121/DisableCopyOnReadRead_121/DisableCopyOnRead22
Read_121/ReadVariableOpRead_121/ReadVariableOp28
Read_122/DisableCopyOnReadRead_122/DisableCopyOnRead22
Read_122/ReadVariableOpRead_122/ReadVariableOp28
Read_123/DisableCopyOnReadRead_123/DisableCopyOnRead22
Read_123/ReadVariableOpRead_123/ReadVariableOp28
Read_124/DisableCopyOnReadRead_124/DisableCopyOnRead22
Read_124/ReadVariableOpRead_124/ReadVariableOp28
Read_125/DisableCopyOnReadRead_125/DisableCopyOnRead22
Read_125/ReadVariableOpRead_125/ReadVariableOp28
Read_126/DisableCopyOnReadRead_126/DisableCopyOnRead22
Read_126/ReadVariableOpRead_126/ReadVariableOp28
Read_127/DisableCopyOnReadRead_127/DisableCopyOnRead22
Read_127/ReadVariableOpRead_127/ReadVariableOp28
Read_128/DisableCopyOnReadRead_128/DisableCopyOnRead22
Read_128/ReadVariableOpRead_128/ReadVariableOp28
Read_129/DisableCopyOnReadRead_129/DisableCopyOnRead22
Read_129/ReadVariableOpRead_129/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp28
Read_130/DisableCopyOnReadRead_130/DisableCopyOnRead22
Read_130/ReadVariableOpRead_130/ReadVariableOp28
Read_131/DisableCopyOnReadRead_131/DisableCopyOnRead22
Read_131/ReadVariableOpRead_131/ReadVariableOp28
Read_132/DisableCopyOnReadRead_132/DisableCopyOnRead22
Read_132/ReadVariableOpRead_132/ReadVariableOp28
Read_133/DisableCopyOnReadRead_133/DisableCopyOnRead22
Read_133/ReadVariableOpRead_133/ReadVariableOp28
Read_134/DisableCopyOnReadRead_134/DisableCopyOnRead22
Read_134/ReadVariableOpRead_134/ReadVariableOp28
Read_135/DisableCopyOnReadRead_135/DisableCopyOnRead22
Read_135/ReadVariableOpRead_135/ReadVariableOp28
Read_136/DisableCopyOnReadRead_136/DisableCopyOnRead22
Read_136/ReadVariableOpRead_136/ReadVariableOp28
Read_137/DisableCopyOnReadRead_137/DisableCopyOnRead22
Read_137/ReadVariableOpRead_137/ReadVariableOp28
Read_138/DisableCopyOnReadRead_138/DisableCopyOnRead22
Read_138/ReadVariableOpRead_138/ReadVariableOp28
Read_139/DisableCopyOnReadRead_139/DisableCopyOnRead22
Read_139/ReadVariableOpRead_139/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp28
Read_140/DisableCopyOnReadRead_140/DisableCopyOnRead22
Read_140/ReadVariableOpRead_140/ReadVariableOp28
Read_141/DisableCopyOnReadRead_141/DisableCopyOnRead22
Read_141/ReadVariableOpRead_141/ReadVariableOp28
Read_142/DisableCopyOnReadRead_142/DisableCopyOnRead22
Read_142/ReadVariableOpRead_142/ReadVariableOp28
Read_143/DisableCopyOnReadRead_143/DisableCopyOnRead22
Read_143/ReadVariableOpRead_143/ReadVariableOp28
Read_144/DisableCopyOnReadRead_144/DisableCopyOnRead22
Read_144/ReadVariableOpRead_144/ReadVariableOp28
Read_145/DisableCopyOnReadRead_145/DisableCopyOnRead22
Read_145/ReadVariableOpRead_145/ReadVariableOp28
Read_146/DisableCopyOnReadRead_146/DisableCopyOnRead22
Read_146/ReadVariableOpRead_146/ReadVariableOp28
Read_147/DisableCopyOnReadRead_147/DisableCopyOnRead22
Read_147/ReadVariableOpRead_147/ReadVariableOp28
Read_148/DisableCopyOnReadRead_148/DisableCopyOnRead22
Read_148/ReadVariableOpRead_148/ReadVariableOp28
Read_149/DisableCopyOnReadRead_149/DisableCopyOnRead22
Read_149/ReadVariableOpRead_149/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp28
Read_150/DisableCopyOnReadRead_150/DisableCopyOnRead22
Read_150/ReadVariableOpRead_150/ReadVariableOp28
Read_151/DisableCopyOnReadRead_151/DisableCopyOnRead22
Read_151/ReadVariableOpRead_151/ReadVariableOp28
Read_152/DisableCopyOnReadRead_152/DisableCopyOnRead22
Read_152/ReadVariableOpRead_152/ReadVariableOp28
Read_153/DisableCopyOnReadRead_153/DisableCopyOnRead22
Read_153/ReadVariableOpRead_153/ReadVariableOp28
Read_154/DisableCopyOnReadRead_154/DisableCopyOnRead22
Read_154/ReadVariableOpRead_154/ReadVariableOp28
Read_155/DisableCopyOnReadRead_155/DisableCopyOnRead22
Read_155/ReadVariableOpRead_155/ReadVariableOp28
Read_156/DisableCopyOnReadRead_156/DisableCopyOnRead22
Read_156/ReadVariableOpRead_156/ReadVariableOp28
Read_157/DisableCopyOnReadRead_157/DisableCopyOnRead22
Read_157/ReadVariableOpRead_157/ReadVariableOp28
Read_158/DisableCopyOnReadRead_158/DisableCopyOnRead22
Read_158/ReadVariableOpRead_158/ReadVariableOp28
Read_159/DisableCopyOnReadRead_159/DisableCopyOnRead22
Read_159/ReadVariableOpRead_159/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp28
Read_160/DisableCopyOnReadRead_160/DisableCopyOnRead22
Read_160/ReadVariableOpRead_160/ReadVariableOp28
Read_161/DisableCopyOnReadRead_161/DisableCopyOnRead22
Read_161/ReadVariableOpRead_161/ReadVariableOp28
Read_162/DisableCopyOnReadRead_162/DisableCopyOnRead22
Read_162/ReadVariableOpRead_162/ReadVariableOp28
Read_163/DisableCopyOnReadRead_163/DisableCopyOnRead22
Read_163/ReadVariableOpRead_163/ReadVariableOp28
Read_164/DisableCopyOnReadRead_164/DisableCopyOnRead22
Read_164/ReadVariableOpRead_164/ReadVariableOp28
Read_165/DisableCopyOnReadRead_165/DisableCopyOnRead22
Read_165/ReadVariableOpRead_165/ReadVariableOp28
Read_166/DisableCopyOnReadRead_166/DisableCopyOnRead22
Read_166/ReadVariableOpRead_166/ReadVariableOp28
Read_167/DisableCopyOnReadRead_167/DisableCopyOnRead22
Read_167/ReadVariableOpRead_167/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp26
Read_80/DisableCopyOnReadRead_80/DisableCopyOnRead20
Read_80/ReadVariableOpRead_80/ReadVariableOp26
Read_81/DisableCopyOnReadRead_81/DisableCopyOnRead20
Read_81/ReadVariableOpRead_81/ReadVariableOp26
Read_82/DisableCopyOnReadRead_82/DisableCopyOnRead20
Read_82/ReadVariableOpRead_82/ReadVariableOp26
Read_83/DisableCopyOnReadRead_83/DisableCopyOnRead20
Read_83/ReadVariableOpRead_83/ReadVariableOp26
Read_84/DisableCopyOnReadRead_84/DisableCopyOnRead20
Read_84/ReadVariableOpRead_84/ReadVariableOp26
Read_85/DisableCopyOnReadRead_85/DisableCopyOnRead20
Read_85/ReadVariableOpRead_85/ReadVariableOp26
Read_86/DisableCopyOnReadRead_86/DisableCopyOnRead20
Read_86/ReadVariableOpRead_86/ReadVariableOp26
Read_87/DisableCopyOnReadRead_87/DisableCopyOnRead20
Read_87/ReadVariableOpRead_87/ReadVariableOp26
Read_88/DisableCopyOnReadRead_88/DisableCopyOnRead20
Read_88/ReadVariableOpRead_88/ReadVariableOp26
Read_89/DisableCopyOnReadRead_89/DisableCopyOnRead20
Read_89/ReadVariableOpRead_89/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp26
Read_90/DisableCopyOnReadRead_90/DisableCopyOnRead20
Read_90/ReadVariableOpRead_90/ReadVariableOp26
Read_91/DisableCopyOnReadRead_91/DisableCopyOnRead20
Read_91/ReadVariableOpRead_91/ReadVariableOp26
Read_92/DisableCopyOnReadRead_92/DisableCopyOnRead20
Read_92/ReadVariableOpRead_92/ReadVariableOp26
Read_93/DisableCopyOnReadRead_93/DisableCopyOnRead20
Read_93/ReadVariableOpRead_93/ReadVariableOp26
Read_94/DisableCopyOnReadRead_94/DisableCopyOnRead20
Read_94/ReadVariableOpRead_94/ReadVariableOp26
Read_95/DisableCopyOnReadRead_95/DisableCopyOnRead20
Read_95/ReadVariableOpRead_95/ReadVariableOp26
Read_96/DisableCopyOnReadRead_96/DisableCopyOnRead20
Read_96/ReadVariableOpRead_96/ReadVariableOp26
Read_97/DisableCopyOnReadRead_97/DisableCopyOnRead20
Read_97/ReadVariableOpRead_97/ReadVariableOp26
Read_98/DisableCopyOnReadRead_98/DisableCopyOnRead20
Read_98/ReadVariableOpRead_98/ReadVariableOp26
Read_99/DisableCopyOnReadRead_99/DisableCopyOnRead20
Read_99/ReadVariableOpRead_99/ReadVariableOp:>�9

_output_shapes
: 

_user_specified_nameConst:&�!

_user_specified_namecount:&�!

_user_specified_nametotal:0�+
)
_user_specified_namev/output_C/bias:0�+
)
_user_specified_namem/output_C/bias:2�-
+
_user_specified_namev/output_C/kernel:2�-
+
_user_specified_namem/output_C/kernel:5�0
.
_user_specified_namev/conv_hid5_5x5/bias:5�0
.
_user_specified_namem/conv_hid5_5x5/bias:7�2
0
_user_specified_namev/conv_hid5_5x5/kernel:7�2
0
_user_specified_namem/conv_hid5_5x5/kernel:5�0
.
_user_specified_namev/conv_hid4_4x4/bias:5�0
.
_user_specified_namem/conv_hid4_4x4/bias:7�2
0
_user_specified_namev/conv_hid4_4x4/kernel:7�2
0
_user_specified_namem/conv_hid4_4x4/kernel:5�0
.
_user_specified_namev/conv_hid4_3x3/bias:5�0
.
_user_specified_namem/conv_hid4_3x3/bias:7�2
0
_user_specified_namev/conv_hid4_3x3/kernel:7�2
0
_user_specified_namem/conv_hid4_3x3/kernel:5�0
.
_user_specified_namev/conv_hid3_5x5/bias:5�0
.
_user_specified_namem/conv_hid3_5x5/bias:7�2
0
_user_specified_namev/conv_hid3_5x5/kernel:7�2
0
_user_specified_namem/conv_hid3_5x5/kernel:5�0
.
_user_specified_namev/conv_hid3_4x4/bias:5�0
.
_user_specified_namem/conv_hid3_4x4/bias:7�2
0
_user_specified_namev/conv_hid3_4x4/kernel:7�2
0
_user_specified_namem/conv_hid3_4x4/kernel:5�0
.
_user_specified_namev/conv_hid3_3x3/bias:5�0
.
_user_specified_namem/conv_hid3_3x3/bias:7�2
0
_user_specified_namev/conv_hid3_3x3/kernel:7�2
0
_user_specified_namem/conv_hid3_3x3/kernel:5�0
.
_user_specified_namev/conv_hid2_5x5/bias:5�0
.
_user_specified_namem/conv_hid2_5x5/bias:7�2
0
_user_specified_namev/conv_hid2_5x5/kernel:7�2
0
_user_specified_namem/conv_hid2_5x5/kernel:5�0
.
_user_specified_namev/conv_hid2_4x4/bias:5�0
.
_user_specified_namem/conv_hid2_4x4/bias:7�2
0
_user_specified_namev/conv_hid2_4x4/kernel:7�2
0
_user_specified_namem/conv_hid2_4x4/kernel:5�0
.
_user_specified_namev/conv_hid2_3x3/bias:5�0
.
_user_specified_namem/conv_hid2_3x3/bias:7�2
0
_user_specified_namev/conv_hid2_3x3/kernel:62
0
_user_specified_namem/conv_hid2_3x3/kernel:4~0
.
_user_specified_namev/conv_hid1_5x5/bias:4}0
.
_user_specified_namem/conv_hid1_5x5/bias:6|2
0
_user_specified_namev/conv_hid1_5x5/kernel:6{2
0
_user_specified_namem/conv_hid1_5x5/kernel:4z0
.
_user_specified_namev/conv_hid1_4x4/bias:4y0
.
_user_specified_namem/conv_hid1_4x4/bias:6x2
0
_user_specified_namev/conv_hid1_4x4/kernel:6w2
0
_user_specified_namem/conv_hid1_4x4/kernel:4v0
.
_user_specified_namev/conv_hid1_3x3/bias:4u0
.
_user_specified_namem/conv_hid1_3x3/bias:6t2
0
_user_specified_namev/conv_hid1_3x3/kernel:6s2
0
_user_specified_namem/conv_hid1_3x3/kernel:4r0
.
_user_specified_namev/conv_hid0_5x5/bias:4q0
.
_user_specified_namem/conv_hid0_5x5/bias:6p2
0
_user_specified_namev/conv_hid0_5x5/kernel:6o2
0
_user_specified_namem/conv_hid0_5x5/kernel:4n0
.
_user_specified_namev/conv_hid0_4x4/bias:4m0
.
_user_specified_namem/conv_hid0_4x4/bias:6l2
0
_user_specified_namev/conv_hid0_4x4/kernel:6k2
0
_user_specified_namem/conv_hid0_4x4/kernel:4j0
.
_user_specified_namev/conv_hid0_3x3/bias:4i0
.
_user_specified_namem/conv_hid0_3x3/bias:6h2
0
_user_specified_namev/conv_hid0_3x3/kernel:6g2
0
_user_specified_namem/conv_hid0_3x3/kernel:5f1
/
_user_specified_namev/conv_prep1_5x5/bias:5e1
/
_user_specified_namem/conv_prep1_5x5/bias:7d3
1
_user_specified_namev/conv_prep1_5x5/kernel:7c3
1
_user_specified_namem/conv_prep1_5x5/kernel:5b1
/
_user_specified_namev/conv_prep1_4x4/bias:5a1
/
_user_specified_namem/conv_prep1_4x4/bias:7`3
1
_user_specified_namev/conv_prep1_4x4/kernel:7_3
1
_user_specified_namem/conv_prep1_4x4/kernel:5^1
/
_user_specified_namev/conv_prep1_3x3/bias:5]1
/
_user_specified_namem/conv_prep1_3x3/bias:7\3
1
_user_specified_namev/conv_prep1_3x3/kernel:7[3
1
_user_specified_namem/conv_prep1_3x3/kernel:5Z1
/
_user_specified_namev/conv_prep0_5x5/bias:5Y1
/
_user_specified_namem/conv_prep0_5x5/bias:7X3
1
_user_specified_namev/conv_prep0_5x5/kernel:7W3
1
_user_specified_namem/conv_prep0_5x5/kernel:5V1
/
_user_specified_namev/conv_prep0_4x4/bias:5U1
/
_user_specified_namem/conv_prep0_4x4/bias:7T3
1
_user_specified_namev/conv_prep0_4x4/kernel:7S3
1
_user_specified_namem/conv_prep0_4x4/kernel:5R1
/
_user_specified_namev/conv_prep0_3x3/bias:5Q1
/
_user_specified_namem/conv_prep0_3x3/bias:7P3
1
_user_specified_namev/conv_prep0_3x3/kernel:7O3
1
_user_specified_namem/conv_prep0_3x3/kernel:-N)
'
_user_specified_namelearning_rate:)M%
#
_user_specified_name	iteration:-L)
'
_user_specified_nameoutput_S/bias:/K+
)
_user_specified_nameoutput_S/kernel:2J.
,
_user_specified_nameconv_rev5_5x5/bias:4I0
.
_user_specified_nameconv_rev5_5x5/kernel:2H.
,
_user_specified_nameconv_rev4_4x4/bias:4G0
.
_user_specified_nameconv_rev4_4x4/kernel:2F.
,
_user_specified_nameconv_rev4_3x3/bias:4E0
.
_user_specified_nameconv_rev4_3x3/kernel:2D.
,
_user_specified_nameconv_rev3_5x5/bias:4C0
.
_user_specified_nameconv_rev3_5x5/kernel:2B.
,
_user_specified_nameconv_rev3_4x4/bias:4A0
.
_user_specified_nameconv_rev3_4x4/kernel:2@.
,
_user_specified_nameconv_rev3_3x3/bias:4?0
.
_user_specified_nameconv_rev3_3x3/kernel:2>.
,
_user_specified_nameconv_rev2_5x5/bias:4=0
.
_user_specified_nameconv_rev2_5x5/kernel:2<.
,
_user_specified_nameconv_rev2_4x4/bias:4;0
.
_user_specified_nameconv_rev2_4x4/kernel:2:.
,
_user_specified_nameconv_rev2_3x3/bias:490
.
_user_specified_nameconv_rev2_3x3/kernel:28.
,
_user_specified_nameconv_rev1_5x5/bias:470
.
_user_specified_nameconv_rev1_5x5/kernel:26.
,
_user_specified_nameconv_rev1_4x4/bias:450
.
_user_specified_nameconv_rev1_4x4/kernel:24.
,
_user_specified_nameconv_rev1_3x3/bias:430
.
_user_specified_nameconv_rev1_3x3/kernel:22.
,
_user_specified_nameconv_rev0_5x5/bias:410
.
_user_specified_nameconv_rev0_5x5/kernel:20.
,
_user_specified_nameconv_rev0_4x4/bias:4/0
.
_user_specified_nameconv_rev0_4x4/kernel:2..
,
_user_specified_nameconv_rev0_3x3/bias:4-0
.
_user_specified_nameconv_rev0_3x3/kernel:-,)
'
_user_specified_nameoutput_C/bias:/++
)
_user_specified_nameoutput_C/kernel:2*.
,
_user_specified_nameconv_hid5_5x5/bias:4)0
.
_user_specified_nameconv_hid5_5x5/kernel:2(.
,
_user_specified_nameconv_hid4_4x4/bias:4'0
.
_user_specified_nameconv_hid4_4x4/kernel:2&.
,
_user_specified_nameconv_hid4_3x3/bias:4%0
.
_user_specified_nameconv_hid4_3x3/kernel:2$.
,
_user_specified_nameconv_hid3_5x5/bias:4#0
.
_user_specified_nameconv_hid3_5x5/kernel:2".
,
_user_specified_nameconv_hid3_4x4/bias:4!0
.
_user_specified_nameconv_hid3_4x4/kernel:2 .
,
_user_specified_nameconv_hid3_3x3/bias:40
.
_user_specified_nameconv_hid3_3x3/kernel:2.
,
_user_specified_nameconv_hid2_5x5/bias:40
.
_user_specified_nameconv_hid2_5x5/kernel:2.
,
_user_specified_nameconv_hid2_4x4/bias:40
.
_user_specified_nameconv_hid2_4x4/kernel:2.
,
_user_specified_nameconv_hid2_3x3/bias:40
.
_user_specified_nameconv_hid2_3x3/kernel:2.
,
_user_specified_nameconv_hid1_5x5/bias:40
.
_user_specified_nameconv_hid1_5x5/kernel:2.
,
_user_specified_nameconv_hid1_4x4/bias:40
.
_user_specified_nameconv_hid1_4x4/kernel:2.
,
_user_specified_nameconv_hid1_3x3/bias:40
.
_user_specified_nameconv_hid1_3x3/kernel:2.
,
_user_specified_nameconv_hid0_5x5/bias:40
.
_user_specified_nameconv_hid0_5x5/kernel:2.
,
_user_specified_nameconv_hid0_4x4/bias:40
.
_user_specified_nameconv_hid0_4x4/kernel:2.
,
_user_specified_nameconv_hid0_3x3/bias:40
.
_user_specified_nameconv_hid0_3x3/kernel:3/
-
_user_specified_nameconv_prep1_5x5/bias:51
/
_user_specified_nameconv_prep1_5x5/kernel:3
/
-
_user_specified_nameconv_prep1_4x4/bias:5	1
/
_user_specified_nameconv_prep1_4x4/kernel:3/
-
_user_specified_nameconv_prep1_3x3/bias:51
/
_user_specified_nameconv_prep1_3x3/kernel:3/
-
_user_specified_nameconv_prep0_5x5/bias:51
/
_user_specified_nameconv_prep0_5x5/kernel:3/
-
_user_specified_nameconv_prep0_4x4/bias:51
/
_user_specified_nameconv_prep0_4x4/kernel:3/
-
_user_specified_nameconv_prep0_3x3/bias:51
/
_user_specified_nameconv_prep0_3x3/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
G__inference_conv_hid4_3x3_layer_call_and_return_conditional_losses_2635

inputs8
conv2d_readvariableop_resource:A2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
,__inference_conv_rev4_3x3_layer_call_fn_5716

inputs!
unknown:A2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev4_3x3_layer_call_and_return_conditional_losses_3516w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@2<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5712:$ 

_user_specified_name5710:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
,__inference_conv_hid4_4x4_layer_call_fn_5316

inputs!
unknown:A

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid4_4x4_layer_call_and_return_conditional_losses_2651w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5312:$ 

_user_specified_name5310:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
f
,__inference_concatenate_9_layer_call_fn_5549
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_9_layer_call_and_return_conditional_losses_3390h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
�
G__inference_conv_rev2_3x3_layer_call_and_return_conditional_losses_3402

inputs8
conv2d_readvariableop_resource:A2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
t
H__inference_concatenate_13_layer_call_and_return_conditional_losses_4824
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������@@:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_0
�
�
,__inference_conv_hid2_3x3_layer_call_fn_5146

inputs!
unknown:A2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid2_3x3_layer_call_and_return_conditional_losses_2521w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@2<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5142:$ 

_user_specified_name5140:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�

G__inference_concatenate_6_layer_call_and_return_conditional_losses_2623

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:WS
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������@@

 
_user_specified_nameinputs:W S
/
_output_shapes
:���������@@2
 
_user_specified_nameinputs
�
�
G__inference_concatenate_4_layer_call_and_return_conditional_losses_5137
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
�
,__inference_conv_rev2_5x5_layer_call_fn_5606

inputs!
unknown:A
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev2_5x5_layer_call_and_return_conditional_losses_3434w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5602:$ 

_user_specified_name5600:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_hid3_4x4_layer_call_and_return_conditional_losses_5252

inputs8
conv2d_readvariableop_resource:A
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_rev0_5x5_layer_call_and_return_conditional_losses_3320

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
G__inference_conv_hid0_5x5_layer_call_and_return_conditional_losses_2439

inputs8
conv2d_readvariableop_resource:D-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:D*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@D: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@D
 
_user_specified_nameinputs
�
�
-__inference_conv_prep1_3x3_layer_call_fn_4908

inputs!
unknown:A2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv_prep1_3x3_layer_call_and_return_conditional_losses_2342w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@2<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name4904:$ 

_user_specified_name4902:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
g
-__inference_concatenate_10_layer_call_fn_5624
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_concatenate_10_layer_call_and_return_conditional_losses_3447h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�

G__inference_concatenate_3_layer_call_and_return_conditional_losses_2452

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:WS
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������@@

 
_user_specified_nameinputs:W S
/
_output_shapes
:���������@@2
 
_user_specified_nameinputs
�
�
G__inference_conv_hid0_4x4_layer_call_and_return_conditional_losses_5027

inputs8
conv2d_readvariableop_resource:D
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:D
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@D: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@D
 
_user_specified_nameinputs
�:
�
$__inference_model_layer_call_fn_4644
input_1
input_2!
unknown:2
	unknown_0:2#
	unknown_1:

	unknown_2:
#
	unknown_3:
	unknown_4:#
	unknown_5:A2
	unknown_6:2#
	unknown_7:A

	unknown_8:
#
	unknown_9:A

unknown_10:$

unknown_11:D2

unknown_12:2$

unknown_13:D


unknown_14:
$

unknown_15:D

unknown_16:$

unknown_17:A2

unknown_18:2$

unknown_19:A


unknown_20:
$

unknown_21:A

unknown_22:$

unknown_23:A2

unknown_24:2$

unknown_25:A


unknown_26:
$

unknown_27:A

unknown_28:$

unknown_29:A2

unknown_30:2$

unknown_31:A


unknown_32:
$

unknown_33:A

unknown_34:$

unknown_35:A2

unknown_36:2$

unknown_37:A


unknown_38:
$

unknown_39:A

unknown_40:$

unknown_41:A

unknown_42:$

unknown_43:2

unknown_44:2$

unknown_45:


unknown_46:
$

unknown_47:

unknown_48:$

unknown_49:A2

unknown_50:2$

unknown_51:A


unknown_52:
$

unknown_53:A

unknown_54:$

unknown_55:A2

unknown_56:2$

unknown_57:A


unknown_58:
$

unknown_59:A

unknown_60:$

unknown_61:A2

unknown_62:2$

unknown_63:A


unknown_64:
$

unknown_65:A

unknown_66:$

unknown_67:A2

unknown_68:2$

unknown_69:A


unknown_70:
$

unknown_71:A

unknown_72:$

unknown_73:A

unknown_74:
identity��StatefulPartitionedCall�

StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74*Y
TinR
P2N*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*n
_read_only_resource_inputsP
NL	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLM*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_4328w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@@:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$M 

_user_specified_name4640:$L 

_user_specified_name4638:$K 

_user_specified_name4636:$J 

_user_specified_name4634:$I 

_user_specified_name4632:$H 

_user_specified_name4630:$G 

_user_specified_name4628:$F 

_user_specified_name4626:$E 

_user_specified_name4624:$D 

_user_specified_name4622:$C 

_user_specified_name4620:$B 

_user_specified_name4618:$A 

_user_specified_name4616:$@ 

_user_specified_name4614:$? 

_user_specified_name4612:$> 

_user_specified_name4610:$= 

_user_specified_name4608:$< 

_user_specified_name4606:$; 

_user_specified_name4604:$: 

_user_specified_name4602:$9 

_user_specified_name4600:$8 

_user_specified_name4598:$7 

_user_specified_name4596:$6 

_user_specified_name4594:$5 

_user_specified_name4592:$4 

_user_specified_name4590:$3 

_user_specified_name4588:$2 

_user_specified_name4586:$1 

_user_specified_name4584:$0 

_user_specified_name4582:$/ 

_user_specified_name4580:$. 

_user_specified_name4578:$- 

_user_specified_name4576:$, 

_user_specified_name4574:$+ 

_user_specified_name4572:$* 

_user_specified_name4570:$) 

_user_specified_name4568:$( 

_user_specified_name4566:$' 

_user_specified_name4564:$& 

_user_specified_name4562:$% 

_user_specified_name4560:$$ 

_user_specified_name4558:$# 

_user_specified_name4556:$" 

_user_specified_name4554:$! 

_user_specified_name4552:$  

_user_specified_name4550:$ 

_user_specified_name4548:$ 

_user_specified_name4546:$ 

_user_specified_name4544:$ 

_user_specified_name4542:$ 

_user_specified_name4540:$ 

_user_specified_name4538:$ 

_user_specified_name4536:$ 

_user_specified_name4534:$ 

_user_specified_name4532:$ 

_user_specified_name4530:$ 

_user_specified_name4528:$ 

_user_specified_name4526:$ 

_user_specified_name4524:$ 

_user_specified_name4522:$ 

_user_specified_name4520:$ 

_user_specified_name4518:$ 

_user_specified_name4516:$ 

_user_specified_name4514:$ 

_user_specified_name4512:$ 

_user_specified_name4510:$ 

_user_specified_name4508:$
 

_user_specified_name4506:$	 

_user_specified_name4504:$ 

_user_specified_name4502:$ 

_user_specified_name4500:$ 

_user_specified_name4498:$ 

_user_specified_name4496:$ 

_user_specified_name4494:$ 

_user_specified_name4492:$ 

_user_specified_name4490:XT
/
_output_shapes
:���������@@
!
_user_specified_name	input_2:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_1
�
g
-__inference_concatenate_11_layer_call_fn_5699
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_concatenate_11_layer_call_and_return_conditional_losses_3504h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
�
G__inference_conv_rev1_5x5_layer_call_and_return_conditional_losses_5542

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
H__inference_conv_prep0_5x5_layer_call_and_return_conditional_losses_4884

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
G__inference_conv_rev0_4x4_layer_call_and_return_conditional_losses_5447

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
B__inference_output_S_layer_call_and_return_conditional_losses_3573

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
۠
�
A__inference_Encoder_layer_call_and_return_conditional_losses_2699
input_3
input_4-
conv_prep0_3x3_2286:2!
conv_prep0_3x3_2288:2-
conv_prep0_4x4_2302:
!
conv_prep0_4x4_2304:
-
conv_prep0_5x5_2318:!
conv_prep0_5x5_2320:-
conv_prep1_3x3_2343:A2!
conv_prep1_3x3_2345:2-
conv_prep1_4x4_2359:A
!
conv_prep1_4x4_2361:
-
conv_prep1_5x5_2375:A!
conv_prep1_5x5_2377:,
conv_hid0_3x3_2408:D2 
conv_hid0_3x3_2410:2,
conv_hid0_4x4_2424:D
 
conv_hid0_4x4_2426:
,
conv_hid0_5x5_2440:D 
conv_hid0_5x5_2442:,
conv_hid1_3x3_2465:A2 
conv_hid1_3x3_2467:2,
conv_hid1_4x4_2481:A
 
conv_hid1_4x4_2483:
,
conv_hid1_5x5_2497:A 
conv_hid1_5x5_2499:,
conv_hid2_3x3_2522:A2 
conv_hid2_3x3_2524:2,
conv_hid2_4x4_2538:A
 
conv_hid2_4x4_2540:
,
conv_hid2_5x5_2554:A 
conv_hid2_5x5_2556:,
conv_hid3_3x3_2579:A2 
conv_hid3_3x3_2581:2,
conv_hid3_4x4_2595:A
 
conv_hid3_4x4_2597:
,
conv_hid3_5x5_2611:A 
conv_hid3_5x5_2613:,
conv_hid4_3x3_2636:A2 
conv_hid4_3x3_2638:2,
conv_hid4_4x4_2652:A
 
conv_hid4_4x4_2654:
,
conv_hid5_5x5_2668:A 
conv_hid5_5x5_2670:'
output_c_2693:A
output_c_2695:
identity��%conv_hid0_3x3/StatefulPartitionedCall�%conv_hid0_4x4/StatefulPartitionedCall�%conv_hid0_5x5/StatefulPartitionedCall�%conv_hid1_3x3/StatefulPartitionedCall�%conv_hid1_4x4/StatefulPartitionedCall�%conv_hid1_5x5/StatefulPartitionedCall�%conv_hid2_3x3/StatefulPartitionedCall�%conv_hid2_4x4/StatefulPartitionedCall�%conv_hid2_5x5/StatefulPartitionedCall�%conv_hid3_3x3/StatefulPartitionedCall�%conv_hid3_4x4/StatefulPartitionedCall�%conv_hid3_5x5/StatefulPartitionedCall�%conv_hid4_3x3/StatefulPartitionedCall�%conv_hid4_4x4/StatefulPartitionedCall�%conv_hid5_5x5/StatefulPartitionedCall�&conv_prep0_3x3/StatefulPartitionedCall�&conv_prep0_4x4/StatefulPartitionedCall�&conv_prep0_5x5/StatefulPartitionedCall�&conv_prep1_3x3/StatefulPartitionedCall�&conv_prep1_4x4/StatefulPartitionedCall�&conv_prep1_5x5/StatefulPartitionedCall� output_C/StatefulPartitionedCall�
&conv_prep0_3x3/StatefulPartitionedCallStatefulPartitionedCallinput_3conv_prep0_3x3_2286conv_prep0_3x3_2288*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv_prep0_3x3_layer_call_and_return_conditional_losses_2285�
&conv_prep0_4x4/StatefulPartitionedCallStatefulPartitionedCallinput_3conv_prep0_4x4_2302conv_prep0_4x4_2304*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv_prep0_4x4_layer_call_and_return_conditional_losses_2301�
&conv_prep0_5x5/StatefulPartitionedCallStatefulPartitionedCallinput_3conv_prep0_5x5_2318conv_prep0_5x5_2320*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv_prep0_5x5_layer_call_and_return_conditional_losses_2317�
concatenate/PartitionedCallPartitionedCall/conv_prep0_3x3/StatefulPartitionedCall:output:0/conv_prep0_4x4/StatefulPartitionedCall:output:0/conv_prep0_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_2330�
&conv_prep1_3x3/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv_prep1_3x3_2343conv_prep1_3x3_2345*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv_prep1_3x3_layer_call_and_return_conditional_losses_2342�
&conv_prep1_4x4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv_prep1_4x4_2359conv_prep1_4x4_2361*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv_prep1_4x4_layer_call_and_return_conditional_losses_2358�
&conv_prep1_5x5/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv_prep1_5x5_2375conv_prep1_5x5_2377*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv_prep1_5x5_layer_call_and_return_conditional_losses_2374�
concatenate_1/PartitionedCallPartitionedCall/conv_prep1_3x3/StatefulPartitionedCall:output:0/conv_prep1_4x4/StatefulPartitionedCall:output:0/conv_prep1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_1_layer_call_and_return_conditional_losses_2387�
concatenate_2/PartitionedCallPartitionedCallinput_4&concatenate_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@D* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_2_layer_call_and_return_conditional_losses_2395�
%conv_hid0_3x3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv_hid0_3x3_2408conv_hid0_3x3_2410*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid0_3x3_layer_call_and_return_conditional_losses_2407�
%conv_hid0_4x4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv_hid0_4x4_2424conv_hid0_4x4_2426*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid0_4x4_layer_call_and_return_conditional_losses_2423�
%conv_hid0_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv_hid0_5x5_2440conv_hid0_5x5_2442*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid0_5x5_layer_call_and_return_conditional_losses_2439�
concatenate_3/PartitionedCallPartitionedCall.conv_hid0_3x3/StatefulPartitionedCall:output:0.conv_hid0_4x4/StatefulPartitionedCall:output:0.conv_hid0_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_3_layer_call_and_return_conditional_losses_2452�
%conv_hid1_3x3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv_hid1_3x3_2465conv_hid1_3x3_2467*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid1_3x3_layer_call_and_return_conditional_losses_2464�
%conv_hid1_4x4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv_hid1_4x4_2481conv_hid1_4x4_2483*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid1_4x4_layer_call_and_return_conditional_losses_2480�
%conv_hid1_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv_hid1_5x5_2497conv_hid1_5x5_2499*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid1_5x5_layer_call_and_return_conditional_losses_2496�
concatenate_4/PartitionedCallPartitionedCall.conv_hid1_3x3/StatefulPartitionedCall:output:0.conv_hid1_4x4/StatefulPartitionedCall:output:0.conv_hid1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_4_layer_call_and_return_conditional_losses_2509�
%conv_hid2_3x3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv_hid2_3x3_2522conv_hid2_3x3_2524*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid2_3x3_layer_call_and_return_conditional_losses_2521�
%conv_hid2_4x4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv_hid2_4x4_2538conv_hid2_4x4_2540*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid2_4x4_layer_call_and_return_conditional_losses_2537�
%conv_hid2_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv_hid2_5x5_2554conv_hid2_5x5_2556*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid2_5x5_layer_call_and_return_conditional_losses_2553�
concatenate_5/PartitionedCallPartitionedCall.conv_hid2_3x3/StatefulPartitionedCall:output:0.conv_hid2_4x4/StatefulPartitionedCall:output:0.conv_hid2_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_5_layer_call_and_return_conditional_losses_2566�
%conv_hid3_3x3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv_hid3_3x3_2579conv_hid3_3x3_2581*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid3_3x3_layer_call_and_return_conditional_losses_2578�
%conv_hid3_4x4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv_hid3_4x4_2595conv_hid3_4x4_2597*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid3_4x4_layer_call_and_return_conditional_losses_2594�
%conv_hid3_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv_hid3_5x5_2611conv_hid3_5x5_2613*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid3_5x5_layer_call_and_return_conditional_losses_2610�
concatenate_6/PartitionedCallPartitionedCall.conv_hid3_3x3/StatefulPartitionedCall:output:0.conv_hid3_4x4/StatefulPartitionedCall:output:0.conv_hid3_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_6_layer_call_and_return_conditional_losses_2623�
%conv_hid4_3x3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_6/PartitionedCall:output:0conv_hid4_3x3_2636conv_hid4_3x3_2638*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid4_3x3_layer_call_and_return_conditional_losses_2635�
%conv_hid4_4x4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_6/PartitionedCall:output:0conv_hid4_4x4_2652conv_hid4_4x4_2654*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid4_4x4_layer_call_and_return_conditional_losses_2651�
%conv_hid5_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_6/PartitionedCall:output:0conv_hid5_5x5_2668conv_hid5_5x5_2670*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid5_5x5_layer_call_and_return_conditional_losses_2667�
concatenate_7/PartitionedCallPartitionedCall.conv_hid4_3x3/StatefulPartitionedCall:output:0.conv_hid4_4x4/StatefulPartitionedCall:output:0.conv_hid5_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_7_layer_call_and_return_conditional_losses_2680�
 output_C/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0output_c_2693output_c_2695*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_output_C_layer_call_and_return_conditional_losses_2692�
IdentityIdentity)output_C/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@�
NoOpNoOp&^conv_hid0_3x3/StatefulPartitionedCall&^conv_hid0_4x4/StatefulPartitionedCall&^conv_hid0_5x5/StatefulPartitionedCall&^conv_hid1_3x3/StatefulPartitionedCall&^conv_hid1_4x4/StatefulPartitionedCall&^conv_hid1_5x5/StatefulPartitionedCall&^conv_hid2_3x3/StatefulPartitionedCall&^conv_hid2_4x4/StatefulPartitionedCall&^conv_hid2_5x5/StatefulPartitionedCall&^conv_hid3_3x3/StatefulPartitionedCall&^conv_hid3_4x4/StatefulPartitionedCall&^conv_hid3_5x5/StatefulPartitionedCall&^conv_hid4_3x3/StatefulPartitionedCall&^conv_hid4_4x4/StatefulPartitionedCall&^conv_hid5_5x5/StatefulPartitionedCall'^conv_prep0_3x3/StatefulPartitionedCall'^conv_prep0_4x4/StatefulPartitionedCall'^conv_prep0_5x5/StatefulPartitionedCall'^conv_prep1_3x3/StatefulPartitionedCall'^conv_prep1_4x4/StatefulPartitionedCall'^conv_prep1_5x5/StatefulPartitionedCall!^output_C/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@@:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%conv_hid0_3x3/StatefulPartitionedCall%conv_hid0_3x3/StatefulPartitionedCall2N
%conv_hid0_4x4/StatefulPartitionedCall%conv_hid0_4x4/StatefulPartitionedCall2N
%conv_hid0_5x5/StatefulPartitionedCall%conv_hid0_5x5/StatefulPartitionedCall2N
%conv_hid1_3x3/StatefulPartitionedCall%conv_hid1_3x3/StatefulPartitionedCall2N
%conv_hid1_4x4/StatefulPartitionedCall%conv_hid1_4x4/StatefulPartitionedCall2N
%conv_hid1_5x5/StatefulPartitionedCall%conv_hid1_5x5/StatefulPartitionedCall2N
%conv_hid2_3x3/StatefulPartitionedCall%conv_hid2_3x3/StatefulPartitionedCall2N
%conv_hid2_4x4/StatefulPartitionedCall%conv_hid2_4x4/StatefulPartitionedCall2N
%conv_hid2_5x5/StatefulPartitionedCall%conv_hid2_5x5/StatefulPartitionedCall2N
%conv_hid3_3x3/StatefulPartitionedCall%conv_hid3_3x3/StatefulPartitionedCall2N
%conv_hid3_4x4/StatefulPartitionedCall%conv_hid3_4x4/StatefulPartitionedCall2N
%conv_hid3_5x5/StatefulPartitionedCall%conv_hid3_5x5/StatefulPartitionedCall2N
%conv_hid4_3x3/StatefulPartitionedCall%conv_hid4_3x3/StatefulPartitionedCall2N
%conv_hid4_4x4/StatefulPartitionedCall%conv_hid4_4x4/StatefulPartitionedCall2N
%conv_hid5_5x5/StatefulPartitionedCall%conv_hid5_5x5/StatefulPartitionedCall2P
&conv_prep0_3x3/StatefulPartitionedCall&conv_prep0_3x3/StatefulPartitionedCall2P
&conv_prep0_4x4/StatefulPartitionedCall&conv_prep0_4x4/StatefulPartitionedCall2P
&conv_prep0_5x5/StatefulPartitionedCall&conv_prep0_5x5/StatefulPartitionedCall2P
&conv_prep1_3x3/StatefulPartitionedCall&conv_prep1_3x3/StatefulPartitionedCall2P
&conv_prep1_4x4/StatefulPartitionedCall&conv_prep1_4x4/StatefulPartitionedCall2P
&conv_prep1_5x5/StatefulPartitionedCall&conv_prep1_5x5/StatefulPartitionedCall2D
 output_C/StatefulPartitionedCall output_C/StatefulPartitionedCall:$- 

_user_specified_name2695:$, 

_user_specified_name2693:$+ 

_user_specified_name2670:$* 

_user_specified_name2668:$) 

_user_specified_name2654:$( 

_user_specified_name2652:$' 

_user_specified_name2638:$& 

_user_specified_name2636:$% 

_user_specified_name2613:$$ 

_user_specified_name2611:$# 

_user_specified_name2597:$" 

_user_specified_name2595:$! 

_user_specified_name2581:$  

_user_specified_name2579:$ 

_user_specified_name2556:$ 

_user_specified_name2554:$ 

_user_specified_name2540:$ 

_user_specified_name2538:$ 

_user_specified_name2524:$ 

_user_specified_name2522:$ 

_user_specified_name2499:$ 

_user_specified_name2497:$ 

_user_specified_name2483:$ 

_user_specified_name2481:$ 

_user_specified_name2467:$ 

_user_specified_name2465:$ 

_user_specified_name2442:$ 

_user_specified_name2440:$ 

_user_specified_name2426:$ 

_user_specified_name2424:$ 

_user_specified_name2410:$ 

_user_specified_name2408:$ 

_user_specified_name2377:$ 

_user_specified_name2375:$ 

_user_specified_name2361:$
 

_user_specified_name2359:$	 

_user_specified_name2345:$ 

_user_specified_name2343:$ 

_user_specified_name2320:$ 

_user_specified_name2318:$ 

_user_specified_name2304:$ 

_user_specified_name2302:$ 

_user_specified_name2288:$ 

_user_specified_name2286:XT
/
_output_shapes
:���������@@
!
_user_specified_name	input_4:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_3
�
�
H__inference_concatenate_10_layer_call_and_return_conditional_losses_5632
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
�
G__inference_conv_rev0_3x3_layer_call_and_return_conditional_losses_3288

inputs8
conv2d_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
s
G__inference_concatenate_2_layer_call_and_return_conditional_losses_4987
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@D_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@D"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������@@:���������@@A:YU
/
_output_shapes
:���������@@A
"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_0
�
�
G__inference_conv_hid1_5x5_layer_call_and_return_conditional_losses_2496

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
��
�O
__inference__wrapped_model_2271
input_1
input_2U
;model_encoder_conv_prep0_3x3_conv2d_readvariableop_resource:2J
<model_encoder_conv_prep0_3x3_biasadd_readvariableop_resource:2U
;model_encoder_conv_prep0_4x4_conv2d_readvariableop_resource:
J
<model_encoder_conv_prep0_4x4_biasadd_readvariableop_resource:
U
;model_encoder_conv_prep0_5x5_conv2d_readvariableop_resource:J
<model_encoder_conv_prep0_5x5_biasadd_readvariableop_resource:U
;model_encoder_conv_prep1_3x3_conv2d_readvariableop_resource:A2J
<model_encoder_conv_prep1_3x3_biasadd_readvariableop_resource:2U
;model_encoder_conv_prep1_4x4_conv2d_readvariableop_resource:A
J
<model_encoder_conv_prep1_4x4_biasadd_readvariableop_resource:
U
;model_encoder_conv_prep1_5x5_conv2d_readvariableop_resource:AJ
<model_encoder_conv_prep1_5x5_biasadd_readvariableop_resource:T
:model_encoder_conv_hid0_3x3_conv2d_readvariableop_resource:D2I
;model_encoder_conv_hid0_3x3_biasadd_readvariableop_resource:2T
:model_encoder_conv_hid0_4x4_conv2d_readvariableop_resource:D
I
;model_encoder_conv_hid0_4x4_biasadd_readvariableop_resource:
T
:model_encoder_conv_hid0_5x5_conv2d_readvariableop_resource:DI
;model_encoder_conv_hid0_5x5_biasadd_readvariableop_resource:T
:model_encoder_conv_hid1_3x3_conv2d_readvariableop_resource:A2I
;model_encoder_conv_hid1_3x3_biasadd_readvariableop_resource:2T
:model_encoder_conv_hid1_4x4_conv2d_readvariableop_resource:A
I
;model_encoder_conv_hid1_4x4_biasadd_readvariableop_resource:
T
:model_encoder_conv_hid1_5x5_conv2d_readvariableop_resource:AI
;model_encoder_conv_hid1_5x5_biasadd_readvariableop_resource:T
:model_encoder_conv_hid2_3x3_conv2d_readvariableop_resource:A2I
;model_encoder_conv_hid2_3x3_biasadd_readvariableop_resource:2T
:model_encoder_conv_hid2_4x4_conv2d_readvariableop_resource:A
I
;model_encoder_conv_hid2_4x4_biasadd_readvariableop_resource:
T
:model_encoder_conv_hid2_5x5_conv2d_readvariableop_resource:AI
;model_encoder_conv_hid2_5x5_biasadd_readvariableop_resource:T
:model_encoder_conv_hid3_3x3_conv2d_readvariableop_resource:A2I
;model_encoder_conv_hid3_3x3_biasadd_readvariableop_resource:2T
:model_encoder_conv_hid3_4x4_conv2d_readvariableop_resource:A
I
;model_encoder_conv_hid3_4x4_biasadd_readvariableop_resource:
T
:model_encoder_conv_hid3_5x5_conv2d_readvariableop_resource:AI
;model_encoder_conv_hid3_5x5_biasadd_readvariableop_resource:T
:model_encoder_conv_hid4_3x3_conv2d_readvariableop_resource:A2I
;model_encoder_conv_hid4_3x3_biasadd_readvariableop_resource:2T
:model_encoder_conv_hid4_4x4_conv2d_readvariableop_resource:A
I
;model_encoder_conv_hid4_4x4_biasadd_readvariableop_resource:
T
:model_encoder_conv_hid5_5x5_conv2d_readvariableop_resource:AI
;model_encoder_conv_hid5_5x5_biasadd_readvariableop_resource:O
5model_encoder_output_c_conv2d_readvariableop_resource:AD
6model_encoder_output_c_biasadd_readvariableop_resource:T
:model_decoder_conv_rev0_3x3_conv2d_readvariableop_resource:2I
;model_decoder_conv_rev0_3x3_biasadd_readvariableop_resource:2T
:model_decoder_conv_rev0_4x4_conv2d_readvariableop_resource:
I
;model_decoder_conv_rev0_4x4_biasadd_readvariableop_resource:
T
:model_decoder_conv_rev0_5x5_conv2d_readvariableop_resource:I
;model_decoder_conv_rev0_5x5_biasadd_readvariableop_resource:T
:model_decoder_conv_rev1_3x3_conv2d_readvariableop_resource:A2I
;model_decoder_conv_rev1_3x3_biasadd_readvariableop_resource:2T
:model_decoder_conv_rev1_4x4_conv2d_readvariableop_resource:A
I
;model_decoder_conv_rev1_4x4_biasadd_readvariableop_resource:
T
:model_decoder_conv_rev1_5x5_conv2d_readvariableop_resource:AI
;model_decoder_conv_rev1_5x5_biasadd_readvariableop_resource:T
:model_decoder_conv_rev2_3x3_conv2d_readvariableop_resource:A2I
;model_decoder_conv_rev2_3x3_biasadd_readvariableop_resource:2T
:model_decoder_conv_rev2_4x4_conv2d_readvariableop_resource:A
I
;model_decoder_conv_rev2_4x4_biasadd_readvariableop_resource:
T
:model_decoder_conv_rev2_5x5_conv2d_readvariableop_resource:AI
;model_decoder_conv_rev2_5x5_biasadd_readvariableop_resource:T
:model_decoder_conv_rev3_3x3_conv2d_readvariableop_resource:A2I
;model_decoder_conv_rev3_3x3_biasadd_readvariableop_resource:2T
:model_decoder_conv_rev3_4x4_conv2d_readvariableop_resource:A
I
;model_decoder_conv_rev3_4x4_biasadd_readvariableop_resource:
T
:model_decoder_conv_rev3_5x5_conv2d_readvariableop_resource:AI
;model_decoder_conv_rev3_5x5_biasadd_readvariableop_resource:T
:model_decoder_conv_rev4_3x3_conv2d_readvariableop_resource:A2I
;model_decoder_conv_rev4_3x3_biasadd_readvariableop_resource:2T
:model_decoder_conv_rev4_4x4_conv2d_readvariableop_resource:A
I
;model_decoder_conv_rev4_4x4_biasadd_readvariableop_resource:
T
:model_decoder_conv_rev5_5x5_conv2d_readvariableop_resource:AI
;model_decoder_conv_rev5_5x5_biasadd_readvariableop_resource:O
5model_decoder_output_s_conv2d_readvariableop_resource:AD
6model_decoder_output_s_biasadd_readvariableop_resource:
identity��2model/Decoder/conv_rev0_3x3/BiasAdd/ReadVariableOp�1model/Decoder/conv_rev0_3x3/Conv2D/ReadVariableOp�2model/Decoder/conv_rev0_4x4/BiasAdd/ReadVariableOp�1model/Decoder/conv_rev0_4x4/Conv2D/ReadVariableOp�2model/Decoder/conv_rev0_5x5/BiasAdd/ReadVariableOp�1model/Decoder/conv_rev0_5x5/Conv2D/ReadVariableOp�2model/Decoder/conv_rev1_3x3/BiasAdd/ReadVariableOp�1model/Decoder/conv_rev1_3x3/Conv2D/ReadVariableOp�2model/Decoder/conv_rev1_4x4/BiasAdd/ReadVariableOp�1model/Decoder/conv_rev1_4x4/Conv2D/ReadVariableOp�2model/Decoder/conv_rev1_5x5/BiasAdd/ReadVariableOp�1model/Decoder/conv_rev1_5x5/Conv2D/ReadVariableOp�2model/Decoder/conv_rev2_3x3/BiasAdd/ReadVariableOp�1model/Decoder/conv_rev2_3x3/Conv2D/ReadVariableOp�2model/Decoder/conv_rev2_4x4/BiasAdd/ReadVariableOp�1model/Decoder/conv_rev2_4x4/Conv2D/ReadVariableOp�2model/Decoder/conv_rev2_5x5/BiasAdd/ReadVariableOp�1model/Decoder/conv_rev2_5x5/Conv2D/ReadVariableOp�2model/Decoder/conv_rev3_3x3/BiasAdd/ReadVariableOp�1model/Decoder/conv_rev3_3x3/Conv2D/ReadVariableOp�2model/Decoder/conv_rev3_4x4/BiasAdd/ReadVariableOp�1model/Decoder/conv_rev3_4x4/Conv2D/ReadVariableOp�2model/Decoder/conv_rev3_5x5/BiasAdd/ReadVariableOp�1model/Decoder/conv_rev3_5x5/Conv2D/ReadVariableOp�2model/Decoder/conv_rev4_3x3/BiasAdd/ReadVariableOp�1model/Decoder/conv_rev4_3x3/Conv2D/ReadVariableOp�2model/Decoder/conv_rev4_4x4/BiasAdd/ReadVariableOp�1model/Decoder/conv_rev4_4x4/Conv2D/ReadVariableOp�2model/Decoder/conv_rev5_5x5/BiasAdd/ReadVariableOp�1model/Decoder/conv_rev5_5x5/Conv2D/ReadVariableOp�-model/Decoder/output_S/BiasAdd/ReadVariableOp�,model/Decoder/output_S/Conv2D/ReadVariableOp�2model/Encoder/conv_hid0_3x3/BiasAdd/ReadVariableOp�1model/Encoder/conv_hid0_3x3/Conv2D/ReadVariableOp�2model/Encoder/conv_hid0_4x4/BiasAdd/ReadVariableOp�1model/Encoder/conv_hid0_4x4/Conv2D/ReadVariableOp�2model/Encoder/conv_hid0_5x5/BiasAdd/ReadVariableOp�1model/Encoder/conv_hid0_5x5/Conv2D/ReadVariableOp�2model/Encoder/conv_hid1_3x3/BiasAdd/ReadVariableOp�1model/Encoder/conv_hid1_3x3/Conv2D/ReadVariableOp�2model/Encoder/conv_hid1_4x4/BiasAdd/ReadVariableOp�1model/Encoder/conv_hid1_4x4/Conv2D/ReadVariableOp�2model/Encoder/conv_hid1_5x5/BiasAdd/ReadVariableOp�1model/Encoder/conv_hid1_5x5/Conv2D/ReadVariableOp�2model/Encoder/conv_hid2_3x3/BiasAdd/ReadVariableOp�1model/Encoder/conv_hid2_3x3/Conv2D/ReadVariableOp�2model/Encoder/conv_hid2_4x4/BiasAdd/ReadVariableOp�1model/Encoder/conv_hid2_4x4/Conv2D/ReadVariableOp�2model/Encoder/conv_hid2_5x5/BiasAdd/ReadVariableOp�1model/Encoder/conv_hid2_5x5/Conv2D/ReadVariableOp�2model/Encoder/conv_hid3_3x3/BiasAdd/ReadVariableOp�1model/Encoder/conv_hid3_3x3/Conv2D/ReadVariableOp�2model/Encoder/conv_hid3_4x4/BiasAdd/ReadVariableOp�1model/Encoder/conv_hid3_4x4/Conv2D/ReadVariableOp�2model/Encoder/conv_hid3_5x5/BiasAdd/ReadVariableOp�1model/Encoder/conv_hid3_5x5/Conv2D/ReadVariableOp�2model/Encoder/conv_hid4_3x3/BiasAdd/ReadVariableOp�1model/Encoder/conv_hid4_3x3/Conv2D/ReadVariableOp�2model/Encoder/conv_hid4_4x4/BiasAdd/ReadVariableOp�1model/Encoder/conv_hid4_4x4/Conv2D/ReadVariableOp�2model/Encoder/conv_hid5_5x5/BiasAdd/ReadVariableOp�1model/Encoder/conv_hid5_5x5/Conv2D/ReadVariableOp�3model/Encoder/conv_prep0_3x3/BiasAdd/ReadVariableOp�2model/Encoder/conv_prep0_3x3/Conv2D/ReadVariableOp�3model/Encoder/conv_prep0_4x4/BiasAdd/ReadVariableOp�2model/Encoder/conv_prep0_4x4/Conv2D/ReadVariableOp�3model/Encoder/conv_prep0_5x5/BiasAdd/ReadVariableOp�2model/Encoder/conv_prep0_5x5/Conv2D/ReadVariableOp�3model/Encoder/conv_prep1_3x3/BiasAdd/ReadVariableOp�2model/Encoder/conv_prep1_3x3/Conv2D/ReadVariableOp�3model/Encoder/conv_prep1_4x4/BiasAdd/ReadVariableOp�2model/Encoder/conv_prep1_4x4/Conv2D/ReadVariableOp�3model/Encoder/conv_prep1_5x5/BiasAdd/ReadVariableOp�2model/Encoder/conv_prep1_5x5/Conv2D/ReadVariableOp�-model/Encoder/output_C/BiasAdd/ReadVariableOp�,model/Encoder/output_C/Conv2D/ReadVariableOp�
2model/Encoder/conv_prep0_3x3/Conv2D/ReadVariableOpReadVariableOp;model_encoder_conv_prep0_3x3_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype0�
#model/Encoder/conv_prep0_3x3/Conv2DConv2Dinput_1:model/Encoder/conv_prep0_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
�
3model/Encoder/conv_prep0_3x3/BiasAdd/ReadVariableOpReadVariableOp<model_encoder_conv_prep0_3x3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
$model/Encoder/conv_prep0_3x3/BiasAddBiasAdd,model/Encoder/conv_prep0_3x3/Conv2D:output:0;model/Encoder/conv_prep0_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2�
!model/Encoder/conv_prep0_3x3/ReluRelu-model/Encoder/conv_prep0_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@2�
2model/Encoder/conv_prep0_4x4/Conv2D/ReadVariableOpReadVariableOp;model_encoder_conv_prep0_4x4_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0�
#model/Encoder/conv_prep0_4x4/Conv2DConv2Dinput_1:model/Encoder/conv_prep0_4x4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
�
3model/Encoder/conv_prep0_4x4/BiasAdd/ReadVariableOpReadVariableOp<model_encoder_conv_prep0_4x4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
$model/Encoder/conv_prep0_4x4/BiasAddBiasAdd,model/Encoder/conv_prep0_4x4/Conv2D:output:0;model/Encoder/conv_prep0_4x4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
�
!model/Encoder/conv_prep0_4x4/ReluRelu-model/Encoder/conv_prep0_4x4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@
�
2model/Encoder/conv_prep0_5x5/Conv2D/ReadVariableOpReadVariableOp;model_encoder_conv_prep0_5x5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
#model/Encoder/conv_prep0_5x5/Conv2DConv2Dinput_1:model/Encoder/conv_prep0_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
3model/Encoder/conv_prep0_5x5/BiasAdd/ReadVariableOpReadVariableOp<model_encoder_conv_prep0_5x5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
$model/Encoder/conv_prep0_5x5/BiasAddBiasAdd,model/Encoder/conv_prep0_5x5/Conv2D:output:0;model/Encoder/conv_prep0_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@�
!model/Encoder/conv_prep0_5x5/ReluRelu-model/Encoder/conv_prep0_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@g
%model/Encoder/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
 model/Encoder/concatenate/concatConcatV2/model/Encoder/conv_prep0_3x3/Relu:activations:0/model/Encoder/conv_prep0_4x4/Relu:activations:0/model/Encoder/conv_prep0_5x5/Relu:activations:0.model/Encoder/concatenate/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A�
2model/Encoder/conv_prep1_3x3/Conv2D/ReadVariableOpReadVariableOp;model_encoder_conv_prep1_3x3_conv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
#model/Encoder/conv_prep1_3x3/Conv2DConv2D)model/Encoder/concatenate/concat:output:0:model/Encoder/conv_prep1_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
�
3model/Encoder/conv_prep1_3x3/BiasAdd/ReadVariableOpReadVariableOp<model_encoder_conv_prep1_3x3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
$model/Encoder/conv_prep1_3x3/BiasAddBiasAdd,model/Encoder/conv_prep1_3x3/Conv2D:output:0;model/Encoder/conv_prep1_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2�
!model/Encoder/conv_prep1_3x3/ReluRelu-model/Encoder/conv_prep1_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@2�
2model/Encoder/conv_prep1_4x4/Conv2D/ReadVariableOpReadVariableOp;model_encoder_conv_prep1_4x4_conv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
#model/Encoder/conv_prep1_4x4/Conv2DConv2D)model/Encoder/concatenate/concat:output:0:model/Encoder/conv_prep1_4x4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
�
3model/Encoder/conv_prep1_4x4/BiasAdd/ReadVariableOpReadVariableOp<model_encoder_conv_prep1_4x4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
$model/Encoder/conv_prep1_4x4/BiasAddBiasAdd,model/Encoder/conv_prep1_4x4/Conv2D:output:0;model/Encoder/conv_prep1_4x4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
�
!model/Encoder/conv_prep1_4x4/ReluRelu-model/Encoder/conv_prep1_4x4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@
�
2model/Encoder/conv_prep1_5x5/Conv2D/ReadVariableOpReadVariableOp;model_encoder_conv_prep1_5x5_conv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
#model/Encoder/conv_prep1_5x5/Conv2DConv2D)model/Encoder/concatenate/concat:output:0:model/Encoder/conv_prep1_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
3model/Encoder/conv_prep1_5x5/BiasAdd/ReadVariableOpReadVariableOp<model_encoder_conv_prep1_5x5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
$model/Encoder/conv_prep1_5x5/BiasAddBiasAdd,model/Encoder/conv_prep1_5x5/Conv2D:output:0;model/Encoder/conv_prep1_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@�
!model/Encoder/conv_prep1_5x5/ReluRelu-model/Encoder/conv_prep1_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
'model/Encoder/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
"model/Encoder/concatenate_1/concatConcatV2/model/Encoder/conv_prep1_3x3/Relu:activations:0/model/Encoder/conv_prep1_4x4/Relu:activations:0/model/Encoder/conv_prep1_5x5/Relu:activations:00model/Encoder/concatenate_1/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@Ai
'model/Encoder/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
"model/Encoder/concatenate_2/concatConcatV2input_2+model/Encoder/concatenate_1/concat:output:00model/Encoder/concatenate_2/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@D�
1model/Encoder/conv_hid0_3x3/Conv2D/ReadVariableOpReadVariableOp:model_encoder_conv_hid0_3x3_conv2d_readvariableop_resource*&
_output_shapes
:D2*
dtype0�
"model/Encoder/conv_hid0_3x3/Conv2DConv2D+model/Encoder/concatenate_2/concat:output:09model/Encoder/conv_hid0_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
�
2model/Encoder/conv_hid0_3x3/BiasAdd/ReadVariableOpReadVariableOp;model_encoder_conv_hid0_3x3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
#model/Encoder/conv_hid0_3x3/BiasAddBiasAdd+model/Encoder/conv_hid0_3x3/Conv2D:output:0:model/Encoder/conv_hid0_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2�
 model/Encoder/conv_hid0_3x3/ReluRelu,model/Encoder/conv_hid0_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@2�
1model/Encoder/conv_hid0_4x4/Conv2D/ReadVariableOpReadVariableOp:model_encoder_conv_hid0_4x4_conv2d_readvariableop_resource*&
_output_shapes
:D
*
dtype0�
"model/Encoder/conv_hid0_4x4/Conv2DConv2D+model/Encoder/concatenate_2/concat:output:09model/Encoder/conv_hid0_4x4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
�
2model/Encoder/conv_hid0_4x4/BiasAdd/ReadVariableOpReadVariableOp;model_encoder_conv_hid0_4x4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
#model/Encoder/conv_hid0_4x4/BiasAddBiasAdd+model/Encoder/conv_hid0_4x4/Conv2D:output:0:model/Encoder/conv_hid0_4x4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
�
 model/Encoder/conv_hid0_4x4/ReluRelu,model/Encoder/conv_hid0_4x4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@
�
1model/Encoder/conv_hid0_5x5/Conv2D/ReadVariableOpReadVariableOp:model_encoder_conv_hid0_5x5_conv2d_readvariableop_resource*&
_output_shapes
:D*
dtype0�
"model/Encoder/conv_hid0_5x5/Conv2DConv2D+model/Encoder/concatenate_2/concat:output:09model/Encoder/conv_hid0_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
2model/Encoder/conv_hid0_5x5/BiasAdd/ReadVariableOpReadVariableOp;model_encoder_conv_hid0_5x5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#model/Encoder/conv_hid0_5x5/BiasAddBiasAdd+model/Encoder/conv_hid0_5x5/Conv2D:output:0:model/Encoder/conv_hid0_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@�
 model/Encoder/conv_hid0_5x5/ReluRelu,model/Encoder/conv_hid0_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
'model/Encoder/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
"model/Encoder/concatenate_3/concatConcatV2.model/Encoder/conv_hid0_3x3/Relu:activations:0.model/Encoder/conv_hid0_4x4/Relu:activations:0.model/Encoder/conv_hid0_5x5/Relu:activations:00model/Encoder/concatenate_3/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A�
1model/Encoder/conv_hid1_3x3/Conv2D/ReadVariableOpReadVariableOp:model_encoder_conv_hid1_3x3_conv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
"model/Encoder/conv_hid1_3x3/Conv2DConv2D+model/Encoder/concatenate_3/concat:output:09model/Encoder/conv_hid1_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
�
2model/Encoder/conv_hid1_3x3/BiasAdd/ReadVariableOpReadVariableOp;model_encoder_conv_hid1_3x3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
#model/Encoder/conv_hid1_3x3/BiasAddBiasAdd+model/Encoder/conv_hid1_3x3/Conv2D:output:0:model/Encoder/conv_hid1_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2�
 model/Encoder/conv_hid1_3x3/ReluRelu,model/Encoder/conv_hid1_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@2�
1model/Encoder/conv_hid1_4x4/Conv2D/ReadVariableOpReadVariableOp:model_encoder_conv_hid1_4x4_conv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
"model/Encoder/conv_hid1_4x4/Conv2DConv2D+model/Encoder/concatenate_3/concat:output:09model/Encoder/conv_hid1_4x4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
�
2model/Encoder/conv_hid1_4x4/BiasAdd/ReadVariableOpReadVariableOp;model_encoder_conv_hid1_4x4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
#model/Encoder/conv_hid1_4x4/BiasAddBiasAdd+model/Encoder/conv_hid1_4x4/Conv2D:output:0:model/Encoder/conv_hid1_4x4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
�
 model/Encoder/conv_hid1_4x4/ReluRelu,model/Encoder/conv_hid1_4x4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@
�
1model/Encoder/conv_hid1_5x5/Conv2D/ReadVariableOpReadVariableOp:model_encoder_conv_hid1_5x5_conv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
"model/Encoder/conv_hid1_5x5/Conv2DConv2D+model/Encoder/concatenate_3/concat:output:09model/Encoder/conv_hid1_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
2model/Encoder/conv_hid1_5x5/BiasAdd/ReadVariableOpReadVariableOp;model_encoder_conv_hid1_5x5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#model/Encoder/conv_hid1_5x5/BiasAddBiasAdd+model/Encoder/conv_hid1_5x5/Conv2D:output:0:model/Encoder/conv_hid1_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@�
 model/Encoder/conv_hid1_5x5/ReluRelu,model/Encoder/conv_hid1_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
'model/Encoder/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
"model/Encoder/concatenate_4/concatConcatV2.model/Encoder/conv_hid1_3x3/Relu:activations:0.model/Encoder/conv_hid1_4x4/Relu:activations:0.model/Encoder/conv_hid1_5x5/Relu:activations:00model/Encoder/concatenate_4/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A�
1model/Encoder/conv_hid2_3x3/Conv2D/ReadVariableOpReadVariableOp:model_encoder_conv_hid2_3x3_conv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
"model/Encoder/conv_hid2_3x3/Conv2DConv2D+model/Encoder/concatenate_4/concat:output:09model/Encoder/conv_hid2_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
�
2model/Encoder/conv_hid2_3x3/BiasAdd/ReadVariableOpReadVariableOp;model_encoder_conv_hid2_3x3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
#model/Encoder/conv_hid2_3x3/BiasAddBiasAdd+model/Encoder/conv_hid2_3x3/Conv2D:output:0:model/Encoder/conv_hid2_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2�
 model/Encoder/conv_hid2_3x3/ReluRelu,model/Encoder/conv_hid2_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@2�
1model/Encoder/conv_hid2_4x4/Conv2D/ReadVariableOpReadVariableOp:model_encoder_conv_hid2_4x4_conv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
"model/Encoder/conv_hid2_4x4/Conv2DConv2D+model/Encoder/concatenate_4/concat:output:09model/Encoder/conv_hid2_4x4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
�
2model/Encoder/conv_hid2_4x4/BiasAdd/ReadVariableOpReadVariableOp;model_encoder_conv_hid2_4x4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
#model/Encoder/conv_hid2_4x4/BiasAddBiasAdd+model/Encoder/conv_hid2_4x4/Conv2D:output:0:model/Encoder/conv_hid2_4x4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
�
 model/Encoder/conv_hid2_4x4/ReluRelu,model/Encoder/conv_hid2_4x4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@
�
1model/Encoder/conv_hid2_5x5/Conv2D/ReadVariableOpReadVariableOp:model_encoder_conv_hid2_5x5_conv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
"model/Encoder/conv_hid2_5x5/Conv2DConv2D+model/Encoder/concatenate_4/concat:output:09model/Encoder/conv_hid2_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
2model/Encoder/conv_hid2_5x5/BiasAdd/ReadVariableOpReadVariableOp;model_encoder_conv_hid2_5x5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#model/Encoder/conv_hid2_5x5/BiasAddBiasAdd+model/Encoder/conv_hid2_5x5/Conv2D:output:0:model/Encoder/conv_hid2_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@�
 model/Encoder/conv_hid2_5x5/ReluRelu,model/Encoder/conv_hid2_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
'model/Encoder/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
"model/Encoder/concatenate_5/concatConcatV2.model/Encoder/conv_hid2_3x3/Relu:activations:0.model/Encoder/conv_hid2_4x4/Relu:activations:0.model/Encoder/conv_hid2_5x5/Relu:activations:00model/Encoder/concatenate_5/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A�
1model/Encoder/conv_hid3_3x3/Conv2D/ReadVariableOpReadVariableOp:model_encoder_conv_hid3_3x3_conv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
"model/Encoder/conv_hid3_3x3/Conv2DConv2D+model/Encoder/concatenate_5/concat:output:09model/Encoder/conv_hid3_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
�
2model/Encoder/conv_hid3_3x3/BiasAdd/ReadVariableOpReadVariableOp;model_encoder_conv_hid3_3x3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
#model/Encoder/conv_hid3_3x3/BiasAddBiasAdd+model/Encoder/conv_hid3_3x3/Conv2D:output:0:model/Encoder/conv_hid3_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2�
 model/Encoder/conv_hid3_3x3/ReluRelu,model/Encoder/conv_hid3_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@2�
1model/Encoder/conv_hid3_4x4/Conv2D/ReadVariableOpReadVariableOp:model_encoder_conv_hid3_4x4_conv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
"model/Encoder/conv_hid3_4x4/Conv2DConv2D+model/Encoder/concatenate_5/concat:output:09model/Encoder/conv_hid3_4x4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
�
2model/Encoder/conv_hid3_4x4/BiasAdd/ReadVariableOpReadVariableOp;model_encoder_conv_hid3_4x4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
#model/Encoder/conv_hid3_4x4/BiasAddBiasAdd+model/Encoder/conv_hid3_4x4/Conv2D:output:0:model/Encoder/conv_hid3_4x4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
�
 model/Encoder/conv_hid3_4x4/ReluRelu,model/Encoder/conv_hid3_4x4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@
�
1model/Encoder/conv_hid3_5x5/Conv2D/ReadVariableOpReadVariableOp:model_encoder_conv_hid3_5x5_conv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
"model/Encoder/conv_hid3_5x5/Conv2DConv2D+model/Encoder/concatenate_5/concat:output:09model/Encoder/conv_hid3_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
2model/Encoder/conv_hid3_5x5/BiasAdd/ReadVariableOpReadVariableOp;model_encoder_conv_hid3_5x5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#model/Encoder/conv_hid3_5x5/BiasAddBiasAdd+model/Encoder/conv_hid3_5x5/Conv2D:output:0:model/Encoder/conv_hid3_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@�
 model/Encoder/conv_hid3_5x5/ReluRelu,model/Encoder/conv_hid3_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
'model/Encoder/concatenate_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
"model/Encoder/concatenate_6/concatConcatV2.model/Encoder/conv_hid3_3x3/Relu:activations:0.model/Encoder/conv_hid3_4x4/Relu:activations:0.model/Encoder/conv_hid3_5x5/Relu:activations:00model/Encoder/concatenate_6/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A�
1model/Encoder/conv_hid4_3x3/Conv2D/ReadVariableOpReadVariableOp:model_encoder_conv_hid4_3x3_conv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
"model/Encoder/conv_hid4_3x3/Conv2DConv2D+model/Encoder/concatenate_6/concat:output:09model/Encoder/conv_hid4_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
�
2model/Encoder/conv_hid4_3x3/BiasAdd/ReadVariableOpReadVariableOp;model_encoder_conv_hid4_3x3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
#model/Encoder/conv_hid4_3x3/BiasAddBiasAdd+model/Encoder/conv_hid4_3x3/Conv2D:output:0:model/Encoder/conv_hid4_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2�
 model/Encoder/conv_hid4_3x3/ReluRelu,model/Encoder/conv_hid4_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@2�
1model/Encoder/conv_hid4_4x4/Conv2D/ReadVariableOpReadVariableOp:model_encoder_conv_hid4_4x4_conv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
"model/Encoder/conv_hid4_4x4/Conv2DConv2D+model/Encoder/concatenate_6/concat:output:09model/Encoder/conv_hid4_4x4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
�
2model/Encoder/conv_hid4_4x4/BiasAdd/ReadVariableOpReadVariableOp;model_encoder_conv_hid4_4x4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
#model/Encoder/conv_hid4_4x4/BiasAddBiasAdd+model/Encoder/conv_hid4_4x4/Conv2D:output:0:model/Encoder/conv_hid4_4x4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
�
 model/Encoder/conv_hid4_4x4/ReluRelu,model/Encoder/conv_hid4_4x4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@
�
1model/Encoder/conv_hid5_5x5/Conv2D/ReadVariableOpReadVariableOp:model_encoder_conv_hid5_5x5_conv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
"model/Encoder/conv_hid5_5x5/Conv2DConv2D+model/Encoder/concatenate_6/concat:output:09model/Encoder/conv_hid5_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
2model/Encoder/conv_hid5_5x5/BiasAdd/ReadVariableOpReadVariableOp;model_encoder_conv_hid5_5x5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#model/Encoder/conv_hid5_5x5/BiasAddBiasAdd+model/Encoder/conv_hid5_5x5/Conv2D:output:0:model/Encoder/conv_hid5_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@�
 model/Encoder/conv_hid5_5x5/ReluRelu,model/Encoder/conv_hid5_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
'model/Encoder/concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
"model/Encoder/concatenate_7/concatConcatV2.model/Encoder/conv_hid4_3x3/Relu:activations:0.model/Encoder/conv_hid4_4x4/Relu:activations:0.model/Encoder/conv_hid5_5x5/Relu:activations:00model/Encoder/concatenate_7/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A�
,model/Encoder/output_C/Conv2D/ReadVariableOpReadVariableOp5model_encoder_output_c_conv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
model/Encoder/output_C/Conv2DConv2D+model/Encoder/concatenate_7/concat:output:04model/Encoder/output_C/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
-model/Encoder/output_C/BiasAdd/ReadVariableOpReadVariableOp6model_encoder_output_c_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/Encoder/output_C/BiasAddBiasAdd&model/Encoder/output_C/Conv2D:output:05model/Encoder/output_C/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@�
model/Encoder/output_C/ReluRelu'model/Encoder/output_C/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@�
1model/Decoder/conv_rev0_3x3/Conv2D/ReadVariableOpReadVariableOp:model_decoder_conv_rev0_3x3_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype0�
"model/Decoder/conv_rev0_3x3/Conv2DConv2D)model/Encoder/output_C/Relu:activations:09model/Decoder/conv_rev0_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
�
2model/Decoder/conv_rev0_3x3/BiasAdd/ReadVariableOpReadVariableOp;model_decoder_conv_rev0_3x3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
#model/Decoder/conv_rev0_3x3/BiasAddBiasAdd+model/Decoder/conv_rev0_3x3/Conv2D:output:0:model/Decoder/conv_rev0_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2�
 model/Decoder/conv_rev0_3x3/ReluRelu,model/Decoder/conv_rev0_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@2�
1model/Decoder/conv_rev0_4x4/Conv2D/ReadVariableOpReadVariableOp:model_decoder_conv_rev0_4x4_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0�
"model/Decoder/conv_rev0_4x4/Conv2DConv2D)model/Encoder/output_C/Relu:activations:09model/Decoder/conv_rev0_4x4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
�
2model/Decoder/conv_rev0_4x4/BiasAdd/ReadVariableOpReadVariableOp;model_decoder_conv_rev0_4x4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
#model/Decoder/conv_rev0_4x4/BiasAddBiasAdd+model/Decoder/conv_rev0_4x4/Conv2D:output:0:model/Decoder/conv_rev0_4x4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
�
 model/Decoder/conv_rev0_4x4/ReluRelu,model/Decoder/conv_rev0_4x4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@
�
1model/Decoder/conv_rev0_5x5/Conv2D/ReadVariableOpReadVariableOp:model_decoder_conv_rev0_5x5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
"model/Decoder/conv_rev0_5x5/Conv2DConv2D)model/Encoder/output_C/Relu:activations:09model/Decoder/conv_rev0_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
2model/Decoder/conv_rev0_5x5/BiasAdd/ReadVariableOpReadVariableOp;model_decoder_conv_rev0_5x5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#model/Decoder/conv_rev0_5x5/BiasAddBiasAdd+model/Decoder/conv_rev0_5x5/Conv2D:output:0:model/Decoder/conv_rev0_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@�
 model/Decoder/conv_rev0_5x5/ReluRelu,model/Decoder/conv_rev0_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
'model/Decoder/concatenate_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
"model/Decoder/concatenate_8/concatConcatV2.model/Decoder/conv_rev0_3x3/Relu:activations:0.model/Decoder/conv_rev0_4x4/Relu:activations:0.model/Decoder/conv_rev0_5x5/Relu:activations:00model/Decoder/concatenate_8/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A�
1model/Decoder/conv_rev1_3x3/Conv2D/ReadVariableOpReadVariableOp:model_decoder_conv_rev1_3x3_conv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
"model/Decoder/conv_rev1_3x3/Conv2DConv2D+model/Decoder/concatenate_8/concat:output:09model/Decoder/conv_rev1_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
�
2model/Decoder/conv_rev1_3x3/BiasAdd/ReadVariableOpReadVariableOp;model_decoder_conv_rev1_3x3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
#model/Decoder/conv_rev1_3x3/BiasAddBiasAdd+model/Decoder/conv_rev1_3x3/Conv2D:output:0:model/Decoder/conv_rev1_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2�
 model/Decoder/conv_rev1_3x3/ReluRelu,model/Decoder/conv_rev1_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@2�
1model/Decoder/conv_rev1_4x4/Conv2D/ReadVariableOpReadVariableOp:model_decoder_conv_rev1_4x4_conv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
"model/Decoder/conv_rev1_4x4/Conv2DConv2D+model/Decoder/concatenate_8/concat:output:09model/Decoder/conv_rev1_4x4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
�
2model/Decoder/conv_rev1_4x4/BiasAdd/ReadVariableOpReadVariableOp;model_decoder_conv_rev1_4x4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
#model/Decoder/conv_rev1_4x4/BiasAddBiasAdd+model/Decoder/conv_rev1_4x4/Conv2D:output:0:model/Decoder/conv_rev1_4x4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
�
 model/Decoder/conv_rev1_4x4/ReluRelu,model/Decoder/conv_rev1_4x4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@
�
1model/Decoder/conv_rev1_5x5/Conv2D/ReadVariableOpReadVariableOp:model_decoder_conv_rev1_5x5_conv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
"model/Decoder/conv_rev1_5x5/Conv2DConv2D+model/Decoder/concatenate_8/concat:output:09model/Decoder/conv_rev1_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
2model/Decoder/conv_rev1_5x5/BiasAdd/ReadVariableOpReadVariableOp;model_decoder_conv_rev1_5x5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#model/Decoder/conv_rev1_5x5/BiasAddBiasAdd+model/Decoder/conv_rev1_5x5/Conv2D:output:0:model/Decoder/conv_rev1_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@�
 model/Decoder/conv_rev1_5x5/ReluRelu,model/Decoder/conv_rev1_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
'model/Decoder/concatenate_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
"model/Decoder/concatenate_9/concatConcatV2.model/Decoder/conv_rev1_3x3/Relu:activations:0.model/Decoder/conv_rev1_4x4/Relu:activations:0.model/Decoder/conv_rev1_5x5/Relu:activations:00model/Decoder/concatenate_9/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A�
1model/Decoder/conv_rev2_3x3/Conv2D/ReadVariableOpReadVariableOp:model_decoder_conv_rev2_3x3_conv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
"model/Decoder/conv_rev2_3x3/Conv2DConv2D+model/Decoder/concatenate_9/concat:output:09model/Decoder/conv_rev2_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
�
2model/Decoder/conv_rev2_3x3/BiasAdd/ReadVariableOpReadVariableOp;model_decoder_conv_rev2_3x3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
#model/Decoder/conv_rev2_3x3/BiasAddBiasAdd+model/Decoder/conv_rev2_3x3/Conv2D:output:0:model/Decoder/conv_rev2_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2�
 model/Decoder/conv_rev2_3x3/ReluRelu,model/Decoder/conv_rev2_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@2�
1model/Decoder/conv_rev2_4x4/Conv2D/ReadVariableOpReadVariableOp:model_decoder_conv_rev2_4x4_conv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
"model/Decoder/conv_rev2_4x4/Conv2DConv2D+model/Decoder/concatenate_9/concat:output:09model/Decoder/conv_rev2_4x4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
�
2model/Decoder/conv_rev2_4x4/BiasAdd/ReadVariableOpReadVariableOp;model_decoder_conv_rev2_4x4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
#model/Decoder/conv_rev2_4x4/BiasAddBiasAdd+model/Decoder/conv_rev2_4x4/Conv2D:output:0:model/Decoder/conv_rev2_4x4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
�
 model/Decoder/conv_rev2_4x4/ReluRelu,model/Decoder/conv_rev2_4x4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@
�
1model/Decoder/conv_rev2_5x5/Conv2D/ReadVariableOpReadVariableOp:model_decoder_conv_rev2_5x5_conv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
"model/Decoder/conv_rev2_5x5/Conv2DConv2D+model/Decoder/concatenate_9/concat:output:09model/Decoder/conv_rev2_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
2model/Decoder/conv_rev2_5x5/BiasAdd/ReadVariableOpReadVariableOp;model_decoder_conv_rev2_5x5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#model/Decoder/conv_rev2_5x5/BiasAddBiasAdd+model/Decoder/conv_rev2_5x5/Conv2D:output:0:model/Decoder/conv_rev2_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@�
 model/Decoder/conv_rev2_5x5/ReluRelu,model/Decoder/conv_rev2_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@j
(model/Decoder/concatenate_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
#model/Decoder/concatenate_10/concatConcatV2.model/Decoder/conv_rev2_3x3/Relu:activations:0.model/Decoder/conv_rev2_4x4/Relu:activations:0.model/Decoder/conv_rev2_5x5/Relu:activations:01model/Decoder/concatenate_10/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A�
1model/Decoder/conv_rev3_3x3/Conv2D/ReadVariableOpReadVariableOp:model_decoder_conv_rev3_3x3_conv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
"model/Decoder/conv_rev3_3x3/Conv2DConv2D,model/Decoder/concatenate_10/concat:output:09model/Decoder/conv_rev3_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
�
2model/Decoder/conv_rev3_3x3/BiasAdd/ReadVariableOpReadVariableOp;model_decoder_conv_rev3_3x3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
#model/Decoder/conv_rev3_3x3/BiasAddBiasAdd+model/Decoder/conv_rev3_3x3/Conv2D:output:0:model/Decoder/conv_rev3_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2�
 model/Decoder/conv_rev3_3x3/ReluRelu,model/Decoder/conv_rev3_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@2�
1model/Decoder/conv_rev3_4x4/Conv2D/ReadVariableOpReadVariableOp:model_decoder_conv_rev3_4x4_conv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
"model/Decoder/conv_rev3_4x4/Conv2DConv2D,model/Decoder/concatenate_10/concat:output:09model/Decoder/conv_rev3_4x4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
�
2model/Decoder/conv_rev3_4x4/BiasAdd/ReadVariableOpReadVariableOp;model_decoder_conv_rev3_4x4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
#model/Decoder/conv_rev3_4x4/BiasAddBiasAdd+model/Decoder/conv_rev3_4x4/Conv2D:output:0:model/Decoder/conv_rev3_4x4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
�
 model/Decoder/conv_rev3_4x4/ReluRelu,model/Decoder/conv_rev3_4x4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@
�
1model/Decoder/conv_rev3_5x5/Conv2D/ReadVariableOpReadVariableOp:model_decoder_conv_rev3_5x5_conv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
"model/Decoder/conv_rev3_5x5/Conv2DConv2D,model/Decoder/concatenate_10/concat:output:09model/Decoder/conv_rev3_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
2model/Decoder/conv_rev3_5x5/BiasAdd/ReadVariableOpReadVariableOp;model_decoder_conv_rev3_5x5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#model/Decoder/conv_rev3_5x5/BiasAddBiasAdd+model/Decoder/conv_rev3_5x5/Conv2D:output:0:model/Decoder/conv_rev3_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@�
 model/Decoder/conv_rev3_5x5/ReluRelu,model/Decoder/conv_rev3_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@j
(model/Decoder/concatenate_11/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
#model/Decoder/concatenate_11/concatConcatV2.model/Decoder/conv_rev3_3x3/Relu:activations:0.model/Decoder/conv_rev3_4x4/Relu:activations:0.model/Decoder/conv_rev3_5x5/Relu:activations:01model/Decoder/concatenate_11/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A�
1model/Decoder/conv_rev4_3x3/Conv2D/ReadVariableOpReadVariableOp:model_decoder_conv_rev4_3x3_conv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
"model/Decoder/conv_rev4_3x3/Conv2DConv2D,model/Decoder/concatenate_11/concat:output:09model/Decoder/conv_rev4_3x3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
�
2model/Decoder/conv_rev4_3x3/BiasAdd/ReadVariableOpReadVariableOp;model_decoder_conv_rev4_3x3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
#model/Decoder/conv_rev4_3x3/BiasAddBiasAdd+model/Decoder/conv_rev4_3x3/Conv2D:output:0:model/Decoder/conv_rev4_3x3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2�
 model/Decoder/conv_rev4_3x3/ReluRelu,model/Decoder/conv_rev4_3x3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@2�
1model/Decoder/conv_rev4_4x4/Conv2D/ReadVariableOpReadVariableOp:model_decoder_conv_rev4_4x4_conv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
"model/Decoder/conv_rev4_4x4/Conv2DConv2D,model/Decoder/concatenate_11/concat:output:09model/Decoder/conv_rev4_4x4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
�
2model/Decoder/conv_rev4_4x4/BiasAdd/ReadVariableOpReadVariableOp;model_decoder_conv_rev4_4x4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
#model/Decoder/conv_rev4_4x4/BiasAddBiasAdd+model/Decoder/conv_rev4_4x4/Conv2D:output:0:model/Decoder/conv_rev4_4x4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
�
 model/Decoder/conv_rev4_4x4/ReluRelu,model/Decoder/conv_rev4_4x4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@
�
1model/Decoder/conv_rev5_5x5/Conv2D/ReadVariableOpReadVariableOp:model_decoder_conv_rev5_5x5_conv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
"model/Decoder/conv_rev5_5x5/Conv2DConv2D,model/Decoder/concatenate_11/concat:output:09model/Decoder/conv_rev5_5x5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
2model/Decoder/conv_rev5_5x5/BiasAdd/ReadVariableOpReadVariableOp;model_decoder_conv_rev5_5x5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#model/Decoder/conv_rev5_5x5/BiasAddBiasAdd+model/Decoder/conv_rev5_5x5/Conv2D:output:0:model/Decoder/conv_rev5_5x5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@�
 model/Decoder/conv_rev5_5x5/ReluRelu,model/Decoder/conv_rev5_5x5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@j
(model/Decoder/concatenate_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
#model/Decoder/concatenate_12/concatConcatV2.model/Decoder/conv_rev4_3x3/Relu:activations:0.model/Decoder/conv_rev4_4x4/Relu:activations:0.model/Decoder/conv_rev5_5x5/Relu:activations:01model/Decoder/concatenate_12/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A�
,model/Decoder/output_S/Conv2D/ReadVariableOpReadVariableOp5model_decoder_output_s_conv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
model/Decoder/output_S/Conv2DConv2D,model/Decoder/concatenate_12/concat:output:04model/Decoder/output_S/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
-model/Decoder/output_S/BiasAdd/ReadVariableOpReadVariableOp6model_decoder_output_s_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/Decoder/output_S/BiasAddBiasAdd&model/Decoder/output_S/Conv2D:output:05model/Decoder/output_S/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@�
model/Decoder/output_S/ReluRelu'model/Decoder/output_S/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@b
 model/concatenate_13/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concatenate_13/concatConcatV2)model/Decoder/output_S/Relu:activations:0)model/Encoder/output_C/Relu:activations:0)model/concatenate_13/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@{
IdentityIdentity$model/concatenate_13/concat:output:0^NoOp*
T0*/
_output_shapes
:���������@@�
NoOpNoOp3^model/Decoder/conv_rev0_3x3/BiasAdd/ReadVariableOp2^model/Decoder/conv_rev0_3x3/Conv2D/ReadVariableOp3^model/Decoder/conv_rev0_4x4/BiasAdd/ReadVariableOp2^model/Decoder/conv_rev0_4x4/Conv2D/ReadVariableOp3^model/Decoder/conv_rev0_5x5/BiasAdd/ReadVariableOp2^model/Decoder/conv_rev0_5x5/Conv2D/ReadVariableOp3^model/Decoder/conv_rev1_3x3/BiasAdd/ReadVariableOp2^model/Decoder/conv_rev1_3x3/Conv2D/ReadVariableOp3^model/Decoder/conv_rev1_4x4/BiasAdd/ReadVariableOp2^model/Decoder/conv_rev1_4x4/Conv2D/ReadVariableOp3^model/Decoder/conv_rev1_5x5/BiasAdd/ReadVariableOp2^model/Decoder/conv_rev1_5x5/Conv2D/ReadVariableOp3^model/Decoder/conv_rev2_3x3/BiasAdd/ReadVariableOp2^model/Decoder/conv_rev2_3x3/Conv2D/ReadVariableOp3^model/Decoder/conv_rev2_4x4/BiasAdd/ReadVariableOp2^model/Decoder/conv_rev2_4x4/Conv2D/ReadVariableOp3^model/Decoder/conv_rev2_5x5/BiasAdd/ReadVariableOp2^model/Decoder/conv_rev2_5x5/Conv2D/ReadVariableOp3^model/Decoder/conv_rev3_3x3/BiasAdd/ReadVariableOp2^model/Decoder/conv_rev3_3x3/Conv2D/ReadVariableOp3^model/Decoder/conv_rev3_4x4/BiasAdd/ReadVariableOp2^model/Decoder/conv_rev3_4x4/Conv2D/ReadVariableOp3^model/Decoder/conv_rev3_5x5/BiasAdd/ReadVariableOp2^model/Decoder/conv_rev3_5x5/Conv2D/ReadVariableOp3^model/Decoder/conv_rev4_3x3/BiasAdd/ReadVariableOp2^model/Decoder/conv_rev4_3x3/Conv2D/ReadVariableOp3^model/Decoder/conv_rev4_4x4/BiasAdd/ReadVariableOp2^model/Decoder/conv_rev4_4x4/Conv2D/ReadVariableOp3^model/Decoder/conv_rev5_5x5/BiasAdd/ReadVariableOp2^model/Decoder/conv_rev5_5x5/Conv2D/ReadVariableOp.^model/Decoder/output_S/BiasAdd/ReadVariableOp-^model/Decoder/output_S/Conv2D/ReadVariableOp3^model/Encoder/conv_hid0_3x3/BiasAdd/ReadVariableOp2^model/Encoder/conv_hid0_3x3/Conv2D/ReadVariableOp3^model/Encoder/conv_hid0_4x4/BiasAdd/ReadVariableOp2^model/Encoder/conv_hid0_4x4/Conv2D/ReadVariableOp3^model/Encoder/conv_hid0_5x5/BiasAdd/ReadVariableOp2^model/Encoder/conv_hid0_5x5/Conv2D/ReadVariableOp3^model/Encoder/conv_hid1_3x3/BiasAdd/ReadVariableOp2^model/Encoder/conv_hid1_3x3/Conv2D/ReadVariableOp3^model/Encoder/conv_hid1_4x4/BiasAdd/ReadVariableOp2^model/Encoder/conv_hid1_4x4/Conv2D/ReadVariableOp3^model/Encoder/conv_hid1_5x5/BiasAdd/ReadVariableOp2^model/Encoder/conv_hid1_5x5/Conv2D/ReadVariableOp3^model/Encoder/conv_hid2_3x3/BiasAdd/ReadVariableOp2^model/Encoder/conv_hid2_3x3/Conv2D/ReadVariableOp3^model/Encoder/conv_hid2_4x4/BiasAdd/ReadVariableOp2^model/Encoder/conv_hid2_4x4/Conv2D/ReadVariableOp3^model/Encoder/conv_hid2_5x5/BiasAdd/ReadVariableOp2^model/Encoder/conv_hid2_5x5/Conv2D/ReadVariableOp3^model/Encoder/conv_hid3_3x3/BiasAdd/ReadVariableOp2^model/Encoder/conv_hid3_3x3/Conv2D/ReadVariableOp3^model/Encoder/conv_hid3_4x4/BiasAdd/ReadVariableOp2^model/Encoder/conv_hid3_4x4/Conv2D/ReadVariableOp3^model/Encoder/conv_hid3_5x5/BiasAdd/ReadVariableOp2^model/Encoder/conv_hid3_5x5/Conv2D/ReadVariableOp3^model/Encoder/conv_hid4_3x3/BiasAdd/ReadVariableOp2^model/Encoder/conv_hid4_3x3/Conv2D/ReadVariableOp3^model/Encoder/conv_hid4_4x4/BiasAdd/ReadVariableOp2^model/Encoder/conv_hid4_4x4/Conv2D/ReadVariableOp3^model/Encoder/conv_hid5_5x5/BiasAdd/ReadVariableOp2^model/Encoder/conv_hid5_5x5/Conv2D/ReadVariableOp4^model/Encoder/conv_prep0_3x3/BiasAdd/ReadVariableOp3^model/Encoder/conv_prep0_3x3/Conv2D/ReadVariableOp4^model/Encoder/conv_prep0_4x4/BiasAdd/ReadVariableOp3^model/Encoder/conv_prep0_4x4/Conv2D/ReadVariableOp4^model/Encoder/conv_prep0_5x5/BiasAdd/ReadVariableOp3^model/Encoder/conv_prep0_5x5/Conv2D/ReadVariableOp4^model/Encoder/conv_prep1_3x3/BiasAdd/ReadVariableOp3^model/Encoder/conv_prep1_3x3/Conv2D/ReadVariableOp4^model/Encoder/conv_prep1_4x4/BiasAdd/ReadVariableOp3^model/Encoder/conv_prep1_4x4/Conv2D/ReadVariableOp4^model/Encoder/conv_prep1_5x5/BiasAdd/ReadVariableOp3^model/Encoder/conv_prep1_5x5/Conv2D/ReadVariableOp.^model/Encoder/output_C/BiasAdd/ReadVariableOp-^model/Encoder/output_C/Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@@:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2model/Decoder/conv_rev0_3x3/BiasAdd/ReadVariableOp2model/Decoder/conv_rev0_3x3/BiasAdd/ReadVariableOp2f
1model/Decoder/conv_rev0_3x3/Conv2D/ReadVariableOp1model/Decoder/conv_rev0_3x3/Conv2D/ReadVariableOp2h
2model/Decoder/conv_rev0_4x4/BiasAdd/ReadVariableOp2model/Decoder/conv_rev0_4x4/BiasAdd/ReadVariableOp2f
1model/Decoder/conv_rev0_4x4/Conv2D/ReadVariableOp1model/Decoder/conv_rev0_4x4/Conv2D/ReadVariableOp2h
2model/Decoder/conv_rev0_5x5/BiasAdd/ReadVariableOp2model/Decoder/conv_rev0_5x5/BiasAdd/ReadVariableOp2f
1model/Decoder/conv_rev0_5x5/Conv2D/ReadVariableOp1model/Decoder/conv_rev0_5x5/Conv2D/ReadVariableOp2h
2model/Decoder/conv_rev1_3x3/BiasAdd/ReadVariableOp2model/Decoder/conv_rev1_3x3/BiasAdd/ReadVariableOp2f
1model/Decoder/conv_rev1_3x3/Conv2D/ReadVariableOp1model/Decoder/conv_rev1_3x3/Conv2D/ReadVariableOp2h
2model/Decoder/conv_rev1_4x4/BiasAdd/ReadVariableOp2model/Decoder/conv_rev1_4x4/BiasAdd/ReadVariableOp2f
1model/Decoder/conv_rev1_4x4/Conv2D/ReadVariableOp1model/Decoder/conv_rev1_4x4/Conv2D/ReadVariableOp2h
2model/Decoder/conv_rev1_5x5/BiasAdd/ReadVariableOp2model/Decoder/conv_rev1_5x5/BiasAdd/ReadVariableOp2f
1model/Decoder/conv_rev1_5x5/Conv2D/ReadVariableOp1model/Decoder/conv_rev1_5x5/Conv2D/ReadVariableOp2h
2model/Decoder/conv_rev2_3x3/BiasAdd/ReadVariableOp2model/Decoder/conv_rev2_3x3/BiasAdd/ReadVariableOp2f
1model/Decoder/conv_rev2_3x3/Conv2D/ReadVariableOp1model/Decoder/conv_rev2_3x3/Conv2D/ReadVariableOp2h
2model/Decoder/conv_rev2_4x4/BiasAdd/ReadVariableOp2model/Decoder/conv_rev2_4x4/BiasAdd/ReadVariableOp2f
1model/Decoder/conv_rev2_4x4/Conv2D/ReadVariableOp1model/Decoder/conv_rev2_4x4/Conv2D/ReadVariableOp2h
2model/Decoder/conv_rev2_5x5/BiasAdd/ReadVariableOp2model/Decoder/conv_rev2_5x5/BiasAdd/ReadVariableOp2f
1model/Decoder/conv_rev2_5x5/Conv2D/ReadVariableOp1model/Decoder/conv_rev2_5x5/Conv2D/ReadVariableOp2h
2model/Decoder/conv_rev3_3x3/BiasAdd/ReadVariableOp2model/Decoder/conv_rev3_3x3/BiasAdd/ReadVariableOp2f
1model/Decoder/conv_rev3_3x3/Conv2D/ReadVariableOp1model/Decoder/conv_rev3_3x3/Conv2D/ReadVariableOp2h
2model/Decoder/conv_rev3_4x4/BiasAdd/ReadVariableOp2model/Decoder/conv_rev3_4x4/BiasAdd/ReadVariableOp2f
1model/Decoder/conv_rev3_4x4/Conv2D/ReadVariableOp1model/Decoder/conv_rev3_4x4/Conv2D/ReadVariableOp2h
2model/Decoder/conv_rev3_5x5/BiasAdd/ReadVariableOp2model/Decoder/conv_rev3_5x5/BiasAdd/ReadVariableOp2f
1model/Decoder/conv_rev3_5x5/Conv2D/ReadVariableOp1model/Decoder/conv_rev3_5x5/Conv2D/ReadVariableOp2h
2model/Decoder/conv_rev4_3x3/BiasAdd/ReadVariableOp2model/Decoder/conv_rev4_3x3/BiasAdd/ReadVariableOp2f
1model/Decoder/conv_rev4_3x3/Conv2D/ReadVariableOp1model/Decoder/conv_rev4_3x3/Conv2D/ReadVariableOp2h
2model/Decoder/conv_rev4_4x4/BiasAdd/ReadVariableOp2model/Decoder/conv_rev4_4x4/BiasAdd/ReadVariableOp2f
1model/Decoder/conv_rev4_4x4/Conv2D/ReadVariableOp1model/Decoder/conv_rev4_4x4/Conv2D/ReadVariableOp2h
2model/Decoder/conv_rev5_5x5/BiasAdd/ReadVariableOp2model/Decoder/conv_rev5_5x5/BiasAdd/ReadVariableOp2f
1model/Decoder/conv_rev5_5x5/Conv2D/ReadVariableOp1model/Decoder/conv_rev5_5x5/Conv2D/ReadVariableOp2^
-model/Decoder/output_S/BiasAdd/ReadVariableOp-model/Decoder/output_S/BiasAdd/ReadVariableOp2\
,model/Decoder/output_S/Conv2D/ReadVariableOp,model/Decoder/output_S/Conv2D/ReadVariableOp2h
2model/Encoder/conv_hid0_3x3/BiasAdd/ReadVariableOp2model/Encoder/conv_hid0_3x3/BiasAdd/ReadVariableOp2f
1model/Encoder/conv_hid0_3x3/Conv2D/ReadVariableOp1model/Encoder/conv_hid0_3x3/Conv2D/ReadVariableOp2h
2model/Encoder/conv_hid0_4x4/BiasAdd/ReadVariableOp2model/Encoder/conv_hid0_4x4/BiasAdd/ReadVariableOp2f
1model/Encoder/conv_hid0_4x4/Conv2D/ReadVariableOp1model/Encoder/conv_hid0_4x4/Conv2D/ReadVariableOp2h
2model/Encoder/conv_hid0_5x5/BiasAdd/ReadVariableOp2model/Encoder/conv_hid0_5x5/BiasAdd/ReadVariableOp2f
1model/Encoder/conv_hid0_5x5/Conv2D/ReadVariableOp1model/Encoder/conv_hid0_5x5/Conv2D/ReadVariableOp2h
2model/Encoder/conv_hid1_3x3/BiasAdd/ReadVariableOp2model/Encoder/conv_hid1_3x3/BiasAdd/ReadVariableOp2f
1model/Encoder/conv_hid1_3x3/Conv2D/ReadVariableOp1model/Encoder/conv_hid1_3x3/Conv2D/ReadVariableOp2h
2model/Encoder/conv_hid1_4x4/BiasAdd/ReadVariableOp2model/Encoder/conv_hid1_4x4/BiasAdd/ReadVariableOp2f
1model/Encoder/conv_hid1_4x4/Conv2D/ReadVariableOp1model/Encoder/conv_hid1_4x4/Conv2D/ReadVariableOp2h
2model/Encoder/conv_hid1_5x5/BiasAdd/ReadVariableOp2model/Encoder/conv_hid1_5x5/BiasAdd/ReadVariableOp2f
1model/Encoder/conv_hid1_5x5/Conv2D/ReadVariableOp1model/Encoder/conv_hid1_5x5/Conv2D/ReadVariableOp2h
2model/Encoder/conv_hid2_3x3/BiasAdd/ReadVariableOp2model/Encoder/conv_hid2_3x3/BiasAdd/ReadVariableOp2f
1model/Encoder/conv_hid2_3x3/Conv2D/ReadVariableOp1model/Encoder/conv_hid2_3x3/Conv2D/ReadVariableOp2h
2model/Encoder/conv_hid2_4x4/BiasAdd/ReadVariableOp2model/Encoder/conv_hid2_4x4/BiasAdd/ReadVariableOp2f
1model/Encoder/conv_hid2_4x4/Conv2D/ReadVariableOp1model/Encoder/conv_hid2_4x4/Conv2D/ReadVariableOp2h
2model/Encoder/conv_hid2_5x5/BiasAdd/ReadVariableOp2model/Encoder/conv_hid2_5x5/BiasAdd/ReadVariableOp2f
1model/Encoder/conv_hid2_5x5/Conv2D/ReadVariableOp1model/Encoder/conv_hid2_5x5/Conv2D/ReadVariableOp2h
2model/Encoder/conv_hid3_3x3/BiasAdd/ReadVariableOp2model/Encoder/conv_hid3_3x3/BiasAdd/ReadVariableOp2f
1model/Encoder/conv_hid3_3x3/Conv2D/ReadVariableOp1model/Encoder/conv_hid3_3x3/Conv2D/ReadVariableOp2h
2model/Encoder/conv_hid3_4x4/BiasAdd/ReadVariableOp2model/Encoder/conv_hid3_4x4/BiasAdd/ReadVariableOp2f
1model/Encoder/conv_hid3_4x4/Conv2D/ReadVariableOp1model/Encoder/conv_hid3_4x4/Conv2D/ReadVariableOp2h
2model/Encoder/conv_hid3_5x5/BiasAdd/ReadVariableOp2model/Encoder/conv_hid3_5x5/BiasAdd/ReadVariableOp2f
1model/Encoder/conv_hid3_5x5/Conv2D/ReadVariableOp1model/Encoder/conv_hid3_5x5/Conv2D/ReadVariableOp2h
2model/Encoder/conv_hid4_3x3/BiasAdd/ReadVariableOp2model/Encoder/conv_hid4_3x3/BiasAdd/ReadVariableOp2f
1model/Encoder/conv_hid4_3x3/Conv2D/ReadVariableOp1model/Encoder/conv_hid4_3x3/Conv2D/ReadVariableOp2h
2model/Encoder/conv_hid4_4x4/BiasAdd/ReadVariableOp2model/Encoder/conv_hid4_4x4/BiasAdd/ReadVariableOp2f
1model/Encoder/conv_hid4_4x4/Conv2D/ReadVariableOp1model/Encoder/conv_hid4_4x4/Conv2D/ReadVariableOp2h
2model/Encoder/conv_hid5_5x5/BiasAdd/ReadVariableOp2model/Encoder/conv_hid5_5x5/BiasAdd/ReadVariableOp2f
1model/Encoder/conv_hid5_5x5/Conv2D/ReadVariableOp1model/Encoder/conv_hid5_5x5/Conv2D/ReadVariableOp2j
3model/Encoder/conv_prep0_3x3/BiasAdd/ReadVariableOp3model/Encoder/conv_prep0_3x3/BiasAdd/ReadVariableOp2h
2model/Encoder/conv_prep0_3x3/Conv2D/ReadVariableOp2model/Encoder/conv_prep0_3x3/Conv2D/ReadVariableOp2j
3model/Encoder/conv_prep0_4x4/BiasAdd/ReadVariableOp3model/Encoder/conv_prep0_4x4/BiasAdd/ReadVariableOp2h
2model/Encoder/conv_prep0_4x4/Conv2D/ReadVariableOp2model/Encoder/conv_prep0_4x4/Conv2D/ReadVariableOp2j
3model/Encoder/conv_prep0_5x5/BiasAdd/ReadVariableOp3model/Encoder/conv_prep0_5x5/BiasAdd/ReadVariableOp2h
2model/Encoder/conv_prep0_5x5/Conv2D/ReadVariableOp2model/Encoder/conv_prep0_5x5/Conv2D/ReadVariableOp2j
3model/Encoder/conv_prep1_3x3/BiasAdd/ReadVariableOp3model/Encoder/conv_prep1_3x3/BiasAdd/ReadVariableOp2h
2model/Encoder/conv_prep1_3x3/Conv2D/ReadVariableOp2model/Encoder/conv_prep1_3x3/Conv2D/ReadVariableOp2j
3model/Encoder/conv_prep1_4x4/BiasAdd/ReadVariableOp3model/Encoder/conv_prep1_4x4/BiasAdd/ReadVariableOp2h
2model/Encoder/conv_prep1_4x4/Conv2D/ReadVariableOp2model/Encoder/conv_prep1_4x4/Conv2D/ReadVariableOp2j
3model/Encoder/conv_prep1_5x5/BiasAdd/ReadVariableOp3model/Encoder/conv_prep1_5x5/BiasAdd/ReadVariableOp2h
2model/Encoder/conv_prep1_5x5/Conv2D/ReadVariableOp2model/Encoder/conv_prep1_5x5/Conv2D/ReadVariableOp2^
-model/Encoder/output_C/BiasAdd/ReadVariableOp-model/Encoder/output_C/BiasAdd/ReadVariableOp2\
,model/Encoder/output_C/Conv2D/ReadVariableOp,model/Encoder/output_C/Conv2D/ReadVariableOp:(M$
"
_user_specified_name
resource:(L$
"
_user_specified_name
resource:(K$
"
_user_specified_name
resource:(J$
"
_user_specified_name
resource:(I$
"
_user_specified_name
resource:(H$
"
_user_specified_name
resource:(G$
"
_user_specified_name
resource:(F$
"
_user_specified_name
resource:(E$
"
_user_specified_name
resource:(D$
"
_user_specified_name
resource:(C$
"
_user_specified_name
resource:(B$
"
_user_specified_name
resource:(A$
"
_user_specified_name
resource:(@$
"
_user_specified_name
resource:(?$
"
_user_specified_name
resource:(>$
"
_user_specified_name
resource:(=$
"
_user_specified_name
resource:(<$
"
_user_specified_name
resource:(;$
"
_user_specified_name
resource:(:$
"
_user_specified_name
resource:(9$
"
_user_specified_name
resource:(8$
"
_user_specified_name
resource:(7$
"
_user_specified_name
resource:(6$
"
_user_specified_name
resource:(5$
"
_user_specified_name
resource:(4$
"
_user_specified_name
resource:(3$
"
_user_specified_name
resource:(2$
"
_user_specified_name
resource:(1$
"
_user_specified_name
resource:(0$
"
_user_specified_name
resource:(/$
"
_user_specified_name
resource:(.$
"
_user_specified_name
resource:(-$
"
_user_specified_name
resource:(,$
"
_user_specified_name
resource:(+$
"
_user_specified_name
resource:(*$
"
_user_specified_name
resource:()$
"
_user_specified_name
resource:(($
"
_user_specified_name
resource:('$
"
_user_specified_name
resource:(&$
"
_user_specified_name
resource:(%$
"
_user_specified_name
resource:($$
"
_user_specified_name
resource:(#$
"
_user_specified_name
resource:("$
"
_user_specified_name
resource:(!$
"
_user_specified_name
resource:( $
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:XT
/
_output_shapes
:���������@@
!
_user_specified_name	input_2:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_1
�
�
G__inference_concatenate_7_layer_call_and_return_conditional_losses_5362
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
�
,__inference_conv_hid0_5x5_layer_call_fn_5036

inputs!
unknown:D
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid0_5x5_layer_call_and_return_conditional_losses_2439w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@D: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5032:$ 

_user_specified_name5030:W S
/
_output_shapes
:���������@@D
 
_user_specified_nameinputs
�
�
,__inference_conv_hid4_3x3_layer_call_fn_5296

inputs!
unknown:A2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid4_3x3_layer_call_and_return_conditional_losses_2635w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@2<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5292:$ 

_user_specified_name5290:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
H__inference_conv_prep0_4x4_layer_call_and_return_conditional_losses_2301

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
q
G__inference_concatenate_2_layer_call_and_return_conditional_losses_2395

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@D_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@D"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������@@:���������@@A:WS
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
,__inference_conv_hid3_4x4_layer_call_fn_5241

inputs!
unknown:A

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid3_4x4_layer_call_and_return_conditional_losses_2594w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5237:$ 

_user_specified_name5235:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_concatenate_8_layer_call_and_return_conditional_losses_5482
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
I
-__inference_output_C_noise_layer_call_fn_5392

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_output_C_noise_layer_call_and_return_conditional_losses_3586h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@@:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
,__inference_conv_rev2_4x4_layer_call_fn_5586

inputs!
unknown:A

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev2_4x4_layer_call_and_return_conditional_losses_3418w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5582:$ 

_user_specified_name5580:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
,__inference_conv_rev3_4x4_layer_call_fn_5661

inputs!
unknown:A

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev3_4x4_layer_call_and_return_conditional_losses_3475w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5657:$ 

_user_specified_name5655:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_concatenate_6_layer_call_and_return_conditional_losses_5287
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
�
G__inference_conv_hid2_4x4_layer_call_and_return_conditional_losses_2537

inputs8
conv2d_readvariableop_resource:A
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_concatenate_1_layer_call_and_return_conditional_losses_4974
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
�
G__inference_conv_rev1_3x3_layer_call_and_return_conditional_losses_3345

inputs8
conv2d_readvariableop_resource:A2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_hid5_5x5_layer_call_and_return_conditional_losses_2667

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_hid0_4x4_layer_call_and_return_conditional_losses_2423

inputs8
conv2d_readvariableop_resource:D
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:D
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@D: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@D
 
_user_specified_nameinputs
�
�
B__inference_output_C_layer_call_and_return_conditional_losses_2692

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
,__inference_conv_rev0_4x4_layer_call_fn_5436

inputs!
unknown:

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev0_4x4_layer_call_and_return_conditional_losses_3304w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5432:$ 

_user_specified_name5430:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
-__inference_conv_prep0_4x4_layer_call_fn_4853

inputs!
unknown:

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv_prep0_4x4_layer_call_and_return_conditional_losses_2301w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name4849:$ 

_user_specified_name4847:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
,__inference_conv_hid0_4x4_layer_call_fn_5016

inputs!
unknown:D

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid0_4x4_layer_call_and_return_conditional_losses_2423w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@D: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5012:$ 

_user_specified_name5010:W S
/
_output_shapes
:���������@@D
 
_user_specified_nameinputs
�
}
E__inference_concatenate_layer_call_and_return_conditional_losses_2330

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:WS
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������@@

 
_user_specified_nameinputs:W S
/
_output_shapes
:���������@@2
 
_user_specified_nameinputs
�
�
G__inference_conv_hid1_5x5_layer_call_and_return_conditional_losses_5122

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_hid2_5x5_layer_call_and_return_conditional_losses_2553

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
H__inference_concatenate_10_layer_call_and_return_conditional_losses_3447

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:WS
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������@@

 
_user_specified_nameinputs:W S
/
_output_shapes
:���������@@2
 
_user_specified_nameinputs
�
�
,__inference_conv_hid5_5x5_layer_call_fn_5336

inputs!
unknown:A
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid5_5x5_layer_call_and_return_conditional_losses_2667w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5332:$ 

_user_specified_name5330:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
d
H__inference_output_C_noise_layer_call_and_return_conditional_losses_3586

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@@:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
,__inference_conv_rev5_5x5_layer_call_fn_5756

inputs!
unknown:A
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev5_5x5_layer_call_and_return_conditional_losses_3548w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5752:$ 

_user_specified_name5750:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
��
�o
 __inference__traced_restore_7346
file_prefix@
&assignvariableop_conv_prep0_3x3_kernel:24
&assignvariableop_1_conv_prep0_3x3_bias:2B
(assignvariableop_2_conv_prep0_4x4_kernel:
4
&assignvariableop_3_conv_prep0_4x4_bias:
B
(assignvariableop_4_conv_prep0_5x5_kernel:4
&assignvariableop_5_conv_prep0_5x5_bias:B
(assignvariableop_6_conv_prep1_3x3_kernel:A24
&assignvariableop_7_conv_prep1_3x3_bias:2B
(assignvariableop_8_conv_prep1_4x4_kernel:A
4
&assignvariableop_9_conv_prep1_4x4_bias:
C
)assignvariableop_10_conv_prep1_5x5_kernel:A5
'assignvariableop_11_conv_prep1_5x5_bias:B
(assignvariableop_12_conv_hid0_3x3_kernel:D24
&assignvariableop_13_conv_hid0_3x3_bias:2B
(assignvariableop_14_conv_hid0_4x4_kernel:D
4
&assignvariableop_15_conv_hid0_4x4_bias:
B
(assignvariableop_16_conv_hid0_5x5_kernel:D4
&assignvariableop_17_conv_hid0_5x5_bias:B
(assignvariableop_18_conv_hid1_3x3_kernel:A24
&assignvariableop_19_conv_hid1_3x3_bias:2B
(assignvariableop_20_conv_hid1_4x4_kernel:A
4
&assignvariableop_21_conv_hid1_4x4_bias:
B
(assignvariableop_22_conv_hid1_5x5_kernel:A4
&assignvariableop_23_conv_hid1_5x5_bias:B
(assignvariableop_24_conv_hid2_3x3_kernel:A24
&assignvariableop_25_conv_hid2_3x3_bias:2B
(assignvariableop_26_conv_hid2_4x4_kernel:A
4
&assignvariableop_27_conv_hid2_4x4_bias:
B
(assignvariableop_28_conv_hid2_5x5_kernel:A4
&assignvariableop_29_conv_hid2_5x5_bias:B
(assignvariableop_30_conv_hid3_3x3_kernel:A24
&assignvariableop_31_conv_hid3_3x3_bias:2B
(assignvariableop_32_conv_hid3_4x4_kernel:A
4
&assignvariableop_33_conv_hid3_4x4_bias:
B
(assignvariableop_34_conv_hid3_5x5_kernel:A4
&assignvariableop_35_conv_hid3_5x5_bias:B
(assignvariableop_36_conv_hid4_3x3_kernel:A24
&assignvariableop_37_conv_hid4_3x3_bias:2B
(assignvariableop_38_conv_hid4_4x4_kernel:A
4
&assignvariableop_39_conv_hid4_4x4_bias:
B
(assignvariableop_40_conv_hid5_5x5_kernel:A4
&assignvariableop_41_conv_hid5_5x5_bias:=
#assignvariableop_42_output_c_kernel:A/
!assignvariableop_43_output_c_bias:B
(assignvariableop_44_conv_rev0_3x3_kernel:24
&assignvariableop_45_conv_rev0_3x3_bias:2B
(assignvariableop_46_conv_rev0_4x4_kernel:
4
&assignvariableop_47_conv_rev0_4x4_bias:
B
(assignvariableop_48_conv_rev0_5x5_kernel:4
&assignvariableop_49_conv_rev0_5x5_bias:B
(assignvariableop_50_conv_rev1_3x3_kernel:A24
&assignvariableop_51_conv_rev1_3x3_bias:2B
(assignvariableop_52_conv_rev1_4x4_kernel:A
4
&assignvariableop_53_conv_rev1_4x4_bias:
B
(assignvariableop_54_conv_rev1_5x5_kernel:A4
&assignvariableop_55_conv_rev1_5x5_bias:B
(assignvariableop_56_conv_rev2_3x3_kernel:A24
&assignvariableop_57_conv_rev2_3x3_bias:2B
(assignvariableop_58_conv_rev2_4x4_kernel:A
4
&assignvariableop_59_conv_rev2_4x4_bias:
B
(assignvariableop_60_conv_rev2_5x5_kernel:A4
&assignvariableop_61_conv_rev2_5x5_bias:B
(assignvariableop_62_conv_rev3_3x3_kernel:A24
&assignvariableop_63_conv_rev3_3x3_bias:2B
(assignvariableop_64_conv_rev3_4x4_kernel:A
4
&assignvariableop_65_conv_rev3_4x4_bias:
B
(assignvariableop_66_conv_rev3_5x5_kernel:A4
&assignvariableop_67_conv_rev3_5x5_bias:B
(assignvariableop_68_conv_rev4_3x3_kernel:A24
&assignvariableop_69_conv_rev4_3x3_bias:2B
(assignvariableop_70_conv_rev4_4x4_kernel:A
4
&assignvariableop_71_conv_rev4_4x4_bias:
B
(assignvariableop_72_conv_rev5_5x5_kernel:A4
&assignvariableop_73_conv_rev5_5x5_bias:=
#assignvariableop_74_output_s_kernel:A/
!assignvariableop_75_output_s_bias:'
assignvariableop_76_iteration:	 +
!assignvariableop_77_learning_rate: E
+assignvariableop_78_m_conv_prep0_3x3_kernel:2E
+assignvariableop_79_v_conv_prep0_3x3_kernel:27
)assignvariableop_80_m_conv_prep0_3x3_bias:27
)assignvariableop_81_v_conv_prep0_3x3_bias:2E
+assignvariableop_82_m_conv_prep0_4x4_kernel:
E
+assignvariableop_83_v_conv_prep0_4x4_kernel:
7
)assignvariableop_84_m_conv_prep0_4x4_bias:
7
)assignvariableop_85_v_conv_prep0_4x4_bias:
E
+assignvariableop_86_m_conv_prep0_5x5_kernel:E
+assignvariableop_87_v_conv_prep0_5x5_kernel:7
)assignvariableop_88_m_conv_prep0_5x5_bias:7
)assignvariableop_89_v_conv_prep0_5x5_bias:E
+assignvariableop_90_m_conv_prep1_3x3_kernel:A2E
+assignvariableop_91_v_conv_prep1_3x3_kernel:A27
)assignvariableop_92_m_conv_prep1_3x3_bias:27
)assignvariableop_93_v_conv_prep1_3x3_bias:2E
+assignvariableop_94_m_conv_prep1_4x4_kernel:A
E
+assignvariableop_95_v_conv_prep1_4x4_kernel:A
7
)assignvariableop_96_m_conv_prep1_4x4_bias:
7
)assignvariableop_97_v_conv_prep1_4x4_bias:
E
+assignvariableop_98_m_conv_prep1_5x5_kernel:AE
+assignvariableop_99_v_conv_prep1_5x5_kernel:A8
*assignvariableop_100_m_conv_prep1_5x5_bias:8
*assignvariableop_101_v_conv_prep1_5x5_bias:E
+assignvariableop_102_m_conv_hid0_3x3_kernel:D2E
+assignvariableop_103_v_conv_hid0_3x3_kernel:D27
)assignvariableop_104_m_conv_hid0_3x3_bias:27
)assignvariableop_105_v_conv_hid0_3x3_bias:2E
+assignvariableop_106_m_conv_hid0_4x4_kernel:D
E
+assignvariableop_107_v_conv_hid0_4x4_kernel:D
7
)assignvariableop_108_m_conv_hid0_4x4_bias:
7
)assignvariableop_109_v_conv_hid0_4x4_bias:
E
+assignvariableop_110_m_conv_hid0_5x5_kernel:DE
+assignvariableop_111_v_conv_hid0_5x5_kernel:D7
)assignvariableop_112_m_conv_hid0_5x5_bias:7
)assignvariableop_113_v_conv_hid0_5x5_bias:E
+assignvariableop_114_m_conv_hid1_3x3_kernel:A2E
+assignvariableop_115_v_conv_hid1_3x3_kernel:A27
)assignvariableop_116_m_conv_hid1_3x3_bias:27
)assignvariableop_117_v_conv_hid1_3x3_bias:2E
+assignvariableop_118_m_conv_hid1_4x4_kernel:A
E
+assignvariableop_119_v_conv_hid1_4x4_kernel:A
7
)assignvariableop_120_m_conv_hid1_4x4_bias:
7
)assignvariableop_121_v_conv_hid1_4x4_bias:
E
+assignvariableop_122_m_conv_hid1_5x5_kernel:AE
+assignvariableop_123_v_conv_hid1_5x5_kernel:A7
)assignvariableop_124_m_conv_hid1_5x5_bias:7
)assignvariableop_125_v_conv_hid1_5x5_bias:E
+assignvariableop_126_m_conv_hid2_3x3_kernel:A2E
+assignvariableop_127_v_conv_hid2_3x3_kernel:A27
)assignvariableop_128_m_conv_hid2_3x3_bias:27
)assignvariableop_129_v_conv_hid2_3x3_bias:2E
+assignvariableop_130_m_conv_hid2_4x4_kernel:A
E
+assignvariableop_131_v_conv_hid2_4x4_kernel:A
7
)assignvariableop_132_m_conv_hid2_4x4_bias:
7
)assignvariableop_133_v_conv_hid2_4x4_bias:
E
+assignvariableop_134_m_conv_hid2_5x5_kernel:AE
+assignvariableop_135_v_conv_hid2_5x5_kernel:A7
)assignvariableop_136_m_conv_hid2_5x5_bias:7
)assignvariableop_137_v_conv_hid2_5x5_bias:E
+assignvariableop_138_m_conv_hid3_3x3_kernel:A2E
+assignvariableop_139_v_conv_hid3_3x3_kernel:A27
)assignvariableop_140_m_conv_hid3_3x3_bias:27
)assignvariableop_141_v_conv_hid3_3x3_bias:2E
+assignvariableop_142_m_conv_hid3_4x4_kernel:A
E
+assignvariableop_143_v_conv_hid3_4x4_kernel:A
7
)assignvariableop_144_m_conv_hid3_4x4_bias:
7
)assignvariableop_145_v_conv_hid3_4x4_bias:
E
+assignvariableop_146_m_conv_hid3_5x5_kernel:AE
+assignvariableop_147_v_conv_hid3_5x5_kernel:A7
)assignvariableop_148_m_conv_hid3_5x5_bias:7
)assignvariableop_149_v_conv_hid3_5x5_bias:E
+assignvariableop_150_m_conv_hid4_3x3_kernel:A2E
+assignvariableop_151_v_conv_hid4_3x3_kernel:A27
)assignvariableop_152_m_conv_hid4_3x3_bias:27
)assignvariableop_153_v_conv_hid4_3x3_bias:2E
+assignvariableop_154_m_conv_hid4_4x4_kernel:A
E
+assignvariableop_155_v_conv_hid4_4x4_kernel:A
7
)assignvariableop_156_m_conv_hid4_4x4_bias:
7
)assignvariableop_157_v_conv_hid4_4x4_bias:
E
+assignvariableop_158_m_conv_hid5_5x5_kernel:AE
+assignvariableop_159_v_conv_hid5_5x5_kernel:A7
)assignvariableop_160_m_conv_hid5_5x5_bias:7
)assignvariableop_161_v_conv_hid5_5x5_bias:@
&assignvariableop_162_m_output_c_kernel:A@
&assignvariableop_163_v_output_c_kernel:A2
$assignvariableop_164_m_output_c_bias:2
$assignvariableop_165_v_output_c_bias:$
assignvariableop_166_total: $
assignvariableop_167_count: 
identity_169��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_113�AssignVariableOp_114�AssignVariableOp_115�AssignVariableOp_116�AssignVariableOp_117�AssignVariableOp_118�AssignVariableOp_119�AssignVariableOp_12�AssignVariableOp_120�AssignVariableOp_121�AssignVariableOp_122�AssignVariableOp_123�AssignVariableOp_124�AssignVariableOp_125�AssignVariableOp_126�AssignVariableOp_127�AssignVariableOp_128�AssignVariableOp_129�AssignVariableOp_13�AssignVariableOp_130�AssignVariableOp_131�AssignVariableOp_132�AssignVariableOp_133�AssignVariableOp_134�AssignVariableOp_135�AssignVariableOp_136�AssignVariableOp_137�AssignVariableOp_138�AssignVariableOp_139�AssignVariableOp_14�AssignVariableOp_140�AssignVariableOp_141�AssignVariableOp_142�AssignVariableOp_143�AssignVariableOp_144�AssignVariableOp_145�AssignVariableOp_146�AssignVariableOp_147�AssignVariableOp_148�AssignVariableOp_149�AssignVariableOp_15�AssignVariableOp_150�AssignVariableOp_151�AssignVariableOp_152�AssignVariableOp_153�AssignVariableOp_154�AssignVariableOp_155�AssignVariableOp_156�AssignVariableOp_157�AssignVariableOp_158�AssignVariableOp_159�AssignVariableOp_16�AssignVariableOp_160�AssignVariableOp_161�AssignVariableOp_162�AssignVariableOp_163�AssignVariableOp_164�AssignVariableOp_165�AssignVariableOp_166�AssignVariableOp_167�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�>
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�>
value�=B�=�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB'variables/56/.ATTRIBUTES/VARIABLE_VALUEB'variables/57/.ATTRIBUTES/VARIABLE_VALUEB'variables/58/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB'variables/60/.ATTRIBUTES/VARIABLE_VALUEB'variables/61/.ATTRIBUTES/VARIABLE_VALUEB'variables/62/.ATTRIBUTES/VARIABLE_VALUEB'variables/63/.ATTRIBUTES/VARIABLE_VALUEB'variables/64/.ATTRIBUTES/VARIABLE_VALUEB'variables/65/.ATTRIBUTES/VARIABLE_VALUEB'variables/66/.ATTRIBUTES/VARIABLE_VALUEB'variables/67/.ATTRIBUTES/VARIABLE_VALUEB'variables/68/.ATTRIBUTES/VARIABLE_VALUEB'variables/69/.ATTRIBUTES/VARIABLE_VALUEB'variables/70/.ATTRIBUTES/VARIABLE_VALUEB'variables/71/.ATTRIBUTES/VARIABLE_VALUEB'variables/72/.ATTRIBUTES/VARIABLE_VALUEB'variables/73/.ATTRIBUTES/VARIABLE_VALUEB'variables/74/.ATTRIBUTES/VARIABLE_VALUEB'variables/75/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/73/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/74/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/75/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/76/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/77/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/78/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/79/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/80/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/81/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/82/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/83/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/84/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/85/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/86/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/87/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/88/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypes�
�2�	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp&assignvariableop_conv_prep0_3x3_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp&assignvariableop_1_conv_prep0_3x3_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp(assignvariableop_2_conv_prep0_4x4_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp&assignvariableop_3_conv_prep0_4x4_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp(assignvariableop_4_conv_prep0_5x5_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp&assignvariableop_5_conv_prep0_5x5_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp(assignvariableop_6_conv_prep1_3x3_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp&assignvariableop_7_conv_prep1_3x3_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp(assignvariableop_8_conv_prep1_4x4_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp&assignvariableop_9_conv_prep1_4x4_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp)assignvariableop_10_conv_prep1_5x5_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp'assignvariableop_11_conv_prep1_5x5_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp(assignvariableop_12_conv_hid0_3x3_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp&assignvariableop_13_conv_hid0_3x3_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp(assignvariableop_14_conv_hid0_4x4_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp&assignvariableop_15_conv_hid0_4x4_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_conv_hid0_5x5_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp&assignvariableop_17_conv_hid0_5x5_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp(assignvariableop_18_conv_hid1_3x3_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp&assignvariableop_19_conv_hid1_3x3_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp(assignvariableop_20_conv_hid1_4x4_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp&assignvariableop_21_conv_hid1_4x4_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp(assignvariableop_22_conv_hid1_5x5_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp&assignvariableop_23_conv_hid1_5x5_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_conv_hid2_3x3_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp&assignvariableop_25_conv_hid2_3x3_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_conv_hid2_4x4_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp&assignvariableop_27_conv_hid2_4x4_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp(assignvariableop_28_conv_hid2_5x5_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp&assignvariableop_29_conv_hid2_5x5_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp(assignvariableop_30_conv_hid3_3x3_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp&assignvariableop_31_conv_hid3_3x3_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp(assignvariableop_32_conv_hid3_4x4_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp&assignvariableop_33_conv_hid3_4x4_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp(assignvariableop_34_conv_hid3_5x5_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp&assignvariableop_35_conv_hid3_5x5_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp(assignvariableop_36_conv_hid4_3x3_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp&assignvariableop_37_conv_hid4_3x3_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp(assignvariableop_38_conv_hid4_4x4_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp&assignvariableop_39_conv_hid4_4x4_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp(assignvariableop_40_conv_hid5_5x5_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp&assignvariableop_41_conv_hid5_5x5_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp#assignvariableop_42_output_c_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp!assignvariableop_43_output_c_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp(assignvariableop_44_conv_rev0_3x3_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp&assignvariableop_45_conv_rev0_3x3_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp(assignvariableop_46_conv_rev0_4x4_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp&assignvariableop_47_conv_rev0_4x4_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp(assignvariableop_48_conv_rev0_5x5_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp&assignvariableop_49_conv_rev0_5x5_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp(assignvariableop_50_conv_rev1_3x3_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp&assignvariableop_51_conv_rev1_3x3_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp(assignvariableop_52_conv_rev1_4x4_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp&assignvariableop_53_conv_rev1_4x4_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp(assignvariableop_54_conv_rev1_5x5_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp&assignvariableop_55_conv_rev1_5x5_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp(assignvariableop_56_conv_rev2_3x3_kernelIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp&assignvariableop_57_conv_rev2_3x3_biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp(assignvariableop_58_conv_rev2_4x4_kernelIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp&assignvariableop_59_conv_rev2_4x4_biasIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp(assignvariableop_60_conv_rev2_5x5_kernelIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp&assignvariableop_61_conv_rev2_5x5_biasIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp(assignvariableop_62_conv_rev3_3x3_kernelIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp&assignvariableop_63_conv_rev3_3x3_biasIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp(assignvariableop_64_conv_rev3_4x4_kernelIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp&assignvariableop_65_conv_rev3_4x4_biasIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp(assignvariableop_66_conv_rev3_5x5_kernelIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp&assignvariableop_67_conv_rev3_5x5_biasIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp(assignvariableop_68_conv_rev4_3x3_kernelIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp&assignvariableop_69_conv_rev4_3x3_biasIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp(assignvariableop_70_conv_rev4_4x4_kernelIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp&assignvariableop_71_conv_rev4_4x4_biasIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp(assignvariableop_72_conv_rev5_5x5_kernelIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp&assignvariableop_73_conv_rev5_5x5_biasIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp#assignvariableop_74_output_s_kernelIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp!assignvariableop_75_output_s_biasIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_76AssignVariableOpassignvariableop_76_iterationIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp!assignvariableop_77_learning_rateIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp+assignvariableop_78_m_conv_prep0_3x3_kernelIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp+assignvariableop_79_v_conv_prep0_3x3_kernelIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp)assignvariableop_80_m_conv_prep0_3x3_biasIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp)assignvariableop_81_v_conv_prep0_3x3_biasIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp+assignvariableop_82_m_conv_prep0_4x4_kernelIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp+assignvariableop_83_v_conv_prep0_4x4_kernelIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp)assignvariableop_84_m_conv_prep0_4x4_biasIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp)assignvariableop_85_v_conv_prep0_4x4_biasIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp+assignvariableop_86_m_conv_prep0_5x5_kernelIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp+assignvariableop_87_v_conv_prep0_5x5_kernelIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp)assignvariableop_88_m_conv_prep0_5x5_biasIdentity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp)assignvariableop_89_v_conv_prep0_5x5_biasIdentity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp+assignvariableop_90_m_conv_prep1_3x3_kernelIdentity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOp+assignvariableop_91_v_conv_prep1_3x3_kernelIdentity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOp)assignvariableop_92_m_conv_prep1_3x3_biasIdentity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp)assignvariableop_93_v_conv_prep1_3x3_biasIdentity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp+assignvariableop_94_m_conv_prep1_4x4_kernelIdentity_94:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOp+assignvariableop_95_v_conv_prep1_4x4_kernelIdentity_95:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOp)assignvariableop_96_m_conv_prep1_4x4_biasIdentity_96:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOp)assignvariableop_97_v_conv_prep1_4x4_biasIdentity_97:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOp+assignvariableop_98_m_conv_prep1_5x5_kernelIdentity_98:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOp+assignvariableop_99_v_conv_prep1_5x5_kernelIdentity_99:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_100AssignVariableOp*assignvariableop_100_m_conv_prep1_5x5_biasIdentity_100:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_101AssignVariableOp*assignvariableop_101_v_conv_prep1_5x5_biasIdentity_101:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_102AssignVariableOp+assignvariableop_102_m_conv_hid0_3x3_kernelIdentity_102:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_103AssignVariableOp+assignvariableop_103_v_conv_hid0_3x3_kernelIdentity_103:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_104AssignVariableOp)assignvariableop_104_m_conv_hid0_3x3_biasIdentity_104:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_105AssignVariableOp)assignvariableop_105_v_conv_hid0_3x3_biasIdentity_105:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_106AssignVariableOp+assignvariableop_106_m_conv_hid0_4x4_kernelIdentity_106:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_107AssignVariableOp+assignvariableop_107_v_conv_hid0_4x4_kernelIdentity_107:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_108AssignVariableOp)assignvariableop_108_m_conv_hid0_4x4_biasIdentity_108:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_109AssignVariableOp)assignvariableop_109_v_conv_hid0_4x4_biasIdentity_109:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_110AssignVariableOp+assignvariableop_110_m_conv_hid0_5x5_kernelIdentity_110:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_111AssignVariableOp+assignvariableop_111_v_conv_hid0_5x5_kernelIdentity_111:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_112AssignVariableOp)assignvariableop_112_m_conv_hid0_5x5_biasIdentity_112:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_113AssignVariableOp)assignvariableop_113_v_conv_hid0_5x5_biasIdentity_113:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_114AssignVariableOp+assignvariableop_114_m_conv_hid1_3x3_kernelIdentity_114:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_115AssignVariableOp+assignvariableop_115_v_conv_hid1_3x3_kernelIdentity_115:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_116AssignVariableOp)assignvariableop_116_m_conv_hid1_3x3_biasIdentity_116:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_117AssignVariableOp)assignvariableop_117_v_conv_hid1_3x3_biasIdentity_117:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_118AssignVariableOp+assignvariableop_118_m_conv_hid1_4x4_kernelIdentity_118:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_119AssignVariableOp+assignvariableop_119_v_conv_hid1_4x4_kernelIdentity_119:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_120AssignVariableOp)assignvariableop_120_m_conv_hid1_4x4_biasIdentity_120:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_121AssignVariableOp)assignvariableop_121_v_conv_hid1_4x4_biasIdentity_121:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_122AssignVariableOp+assignvariableop_122_m_conv_hid1_5x5_kernelIdentity_122:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_123AssignVariableOp+assignvariableop_123_v_conv_hid1_5x5_kernelIdentity_123:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_124AssignVariableOp)assignvariableop_124_m_conv_hid1_5x5_biasIdentity_124:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_125AssignVariableOp)assignvariableop_125_v_conv_hid1_5x5_biasIdentity_125:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_126AssignVariableOp+assignvariableop_126_m_conv_hid2_3x3_kernelIdentity_126:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_127AssignVariableOp+assignvariableop_127_v_conv_hid2_3x3_kernelIdentity_127:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_128AssignVariableOp)assignvariableop_128_m_conv_hid2_3x3_biasIdentity_128:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_129AssignVariableOp)assignvariableop_129_v_conv_hid2_3x3_biasIdentity_129:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_130AssignVariableOp+assignvariableop_130_m_conv_hid2_4x4_kernelIdentity_130:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_131AssignVariableOp+assignvariableop_131_v_conv_hid2_4x4_kernelIdentity_131:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_132AssignVariableOp)assignvariableop_132_m_conv_hid2_4x4_biasIdentity_132:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_133AssignVariableOp)assignvariableop_133_v_conv_hid2_4x4_biasIdentity_133:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_134AssignVariableOp+assignvariableop_134_m_conv_hid2_5x5_kernelIdentity_134:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_135AssignVariableOp+assignvariableop_135_v_conv_hid2_5x5_kernelIdentity_135:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_136AssignVariableOp)assignvariableop_136_m_conv_hid2_5x5_biasIdentity_136:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_137AssignVariableOp)assignvariableop_137_v_conv_hid2_5x5_biasIdentity_137:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_138AssignVariableOp+assignvariableop_138_m_conv_hid3_3x3_kernelIdentity_138:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_139AssignVariableOp+assignvariableop_139_v_conv_hid3_3x3_kernelIdentity_139:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_140AssignVariableOp)assignvariableop_140_m_conv_hid3_3x3_biasIdentity_140:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_141AssignVariableOp)assignvariableop_141_v_conv_hid3_3x3_biasIdentity_141:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_142AssignVariableOp+assignvariableop_142_m_conv_hid3_4x4_kernelIdentity_142:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_143AssignVariableOp+assignvariableop_143_v_conv_hid3_4x4_kernelIdentity_143:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_144AssignVariableOp)assignvariableop_144_m_conv_hid3_4x4_biasIdentity_144:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_145AssignVariableOp)assignvariableop_145_v_conv_hid3_4x4_biasIdentity_145:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_146AssignVariableOp+assignvariableop_146_m_conv_hid3_5x5_kernelIdentity_146:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_147AssignVariableOp+assignvariableop_147_v_conv_hid3_5x5_kernelIdentity_147:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_148AssignVariableOp)assignvariableop_148_m_conv_hid3_5x5_biasIdentity_148:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_149AssignVariableOp)assignvariableop_149_v_conv_hid3_5x5_biasIdentity_149:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_150AssignVariableOp+assignvariableop_150_m_conv_hid4_3x3_kernelIdentity_150:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_151AssignVariableOp+assignvariableop_151_v_conv_hid4_3x3_kernelIdentity_151:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_152AssignVariableOp)assignvariableop_152_m_conv_hid4_3x3_biasIdentity_152:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_153AssignVariableOp)assignvariableop_153_v_conv_hid4_3x3_biasIdentity_153:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_154IdentityRestoreV2:tensors:154"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_154AssignVariableOp+assignvariableop_154_m_conv_hid4_4x4_kernelIdentity_154:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_155IdentityRestoreV2:tensors:155"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_155AssignVariableOp+assignvariableop_155_v_conv_hid4_4x4_kernelIdentity_155:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_156IdentityRestoreV2:tensors:156"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_156AssignVariableOp)assignvariableop_156_m_conv_hid4_4x4_biasIdentity_156:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_157IdentityRestoreV2:tensors:157"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_157AssignVariableOp)assignvariableop_157_v_conv_hid4_4x4_biasIdentity_157:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_158IdentityRestoreV2:tensors:158"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_158AssignVariableOp+assignvariableop_158_m_conv_hid5_5x5_kernelIdentity_158:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_159IdentityRestoreV2:tensors:159"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_159AssignVariableOp+assignvariableop_159_v_conv_hid5_5x5_kernelIdentity_159:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_160IdentityRestoreV2:tensors:160"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_160AssignVariableOp)assignvariableop_160_m_conv_hid5_5x5_biasIdentity_160:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_161IdentityRestoreV2:tensors:161"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_161AssignVariableOp)assignvariableop_161_v_conv_hid5_5x5_biasIdentity_161:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_162IdentityRestoreV2:tensors:162"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_162AssignVariableOp&assignvariableop_162_m_output_c_kernelIdentity_162:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_163IdentityRestoreV2:tensors:163"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_163AssignVariableOp&assignvariableop_163_v_output_c_kernelIdentity_163:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_164IdentityRestoreV2:tensors:164"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_164AssignVariableOp$assignvariableop_164_m_output_c_biasIdentity_164:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_165IdentityRestoreV2:tensors:165"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_165AssignVariableOp$assignvariableop_165_v_output_c_biasIdentity_165:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_166IdentityRestoreV2:tensors:166"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_166AssignVariableOpassignvariableop_166_totalIdentity_166:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_167IdentityRestoreV2:tensors:167"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_167AssignVariableOpassignvariableop_167_countIdentity_167:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_168Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_169IdentityIdentity_168:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
_output_shapes
 "%
identity_169Identity_169:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422,
AssignVariableOp_143AssignVariableOp_1432,
AssignVariableOp_144AssignVariableOp_1442,
AssignVariableOp_145AssignVariableOp_1452,
AssignVariableOp_146AssignVariableOp_1462,
AssignVariableOp_147AssignVariableOp_1472,
AssignVariableOp_148AssignVariableOp_1482,
AssignVariableOp_149AssignVariableOp_1492*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_150AssignVariableOp_1502,
AssignVariableOp_151AssignVariableOp_1512,
AssignVariableOp_152AssignVariableOp_1522,
AssignVariableOp_153AssignVariableOp_1532,
AssignVariableOp_154AssignVariableOp_1542,
AssignVariableOp_155AssignVariableOp_1552,
AssignVariableOp_156AssignVariableOp_1562,
AssignVariableOp_157AssignVariableOp_1572,
AssignVariableOp_158AssignVariableOp_1582,
AssignVariableOp_159AssignVariableOp_1592*
AssignVariableOp_15AssignVariableOp_152,
AssignVariableOp_160AssignVariableOp_1602,
AssignVariableOp_161AssignVariableOp_1612,
AssignVariableOp_162AssignVariableOp_1622,
AssignVariableOp_163AssignVariableOp_1632,
AssignVariableOp_164AssignVariableOp_1642,
AssignVariableOp_165AssignVariableOp_1652,
AssignVariableOp_166AssignVariableOp_1662,
AssignVariableOp_167AssignVariableOp_1672*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_992(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:&�!

_user_specified_namecount:&�!

_user_specified_nametotal:0�+
)
_user_specified_namev/output_C/bias:0�+
)
_user_specified_namem/output_C/bias:2�-
+
_user_specified_namev/output_C/kernel:2�-
+
_user_specified_namem/output_C/kernel:5�0
.
_user_specified_namev/conv_hid5_5x5/bias:5�0
.
_user_specified_namem/conv_hid5_5x5/bias:7�2
0
_user_specified_namev/conv_hid5_5x5/kernel:7�2
0
_user_specified_namem/conv_hid5_5x5/kernel:5�0
.
_user_specified_namev/conv_hid4_4x4/bias:5�0
.
_user_specified_namem/conv_hid4_4x4/bias:7�2
0
_user_specified_namev/conv_hid4_4x4/kernel:7�2
0
_user_specified_namem/conv_hid4_4x4/kernel:5�0
.
_user_specified_namev/conv_hid4_3x3/bias:5�0
.
_user_specified_namem/conv_hid4_3x3/bias:7�2
0
_user_specified_namev/conv_hid4_3x3/kernel:7�2
0
_user_specified_namem/conv_hid4_3x3/kernel:5�0
.
_user_specified_namev/conv_hid3_5x5/bias:5�0
.
_user_specified_namem/conv_hid3_5x5/bias:7�2
0
_user_specified_namev/conv_hid3_5x5/kernel:7�2
0
_user_specified_namem/conv_hid3_5x5/kernel:5�0
.
_user_specified_namev/conv_hid3_4x4/bias:5�0
.
_user_specified_namem/conv_hid3_4x4/bias:7�2
0
_user_specified_namev/conv_hid3_4x4/kernel:7�2
0
_user_specified_namem/conv_hid3_4x4/kernel:5�0
.
_user_specified_namev/conv_hid3_3x3/bias:5�0
.
_user_specified_namem/conv_hid3_3x3/bias:7�2
0
_user_specified_namev/conv_hid3_3x3/kernel:7�2
0
_user_specified_namem/conv_hid3_3x3/kernel:5�0
.
_user_specified_namev/conv_hid2_5x5/bias:5�0
.
_user_specified_namem/conv_hid2_5x5/bias:7�2
0
_user_specified_namev/conv_hid2_5x5/kernel:7�2
0
_user_specified_namem/conv_hid2_5x5/kernel:5�0
.
_user_specified_namev/conv_hid2_4x4/bias:5�0
.
_user_specified_namem/conv_hid2_4x4/bias:7�2
0
_user_specified_namev/conv_hid2_4x4/kernel:7�2
0
_user_specified_namem/conv_hid2_4x4/kernel:5�0
.
_user_specified_namev/conv_hid2_3x3/bias:5�0
.
_user_specified_namem/conv_hid2_3x3/bias:7�2
0
_user_specified_namev/conv_hid2_3x3/kernel:62
0
_user_specified_namem/conv_hid2_3x3/kernel:4~0
.
_user_specified_namev/conv_hid1_5x5/bias:4}0
.
_user_specified_namem/conv_hid1_5x5/bias:6|2
0
_user_specified_namev/conv_hid1_5x5/kernel:6{2
0
_user_specified_namem/conv_hid1_5x5/kernel:4z0
.
_user_specified_namev/conv_hid1_4x4/bias:4y0
.
_user_specified_namem/conv_hid1_4x4/bias:6x2
0
_user_specified_namev/conv_hid1_4x4/kernel:6w2
0
_user_specified_namem/conv_hid1_4x4/kernel:4v0
.
_user_specified_namev/conv_hid1_3x3/bias:4u0
.
_user_specified_namem/conv_hid1_3x3/bias:6t2
0
_user_specified_namev/conv_hid1_3x3/kernel:6s2
0
_user_specified_namem/conv_hid1_3x3/kernel:4r0
.
_user_specified_namev/conv_hid0_5x5/bias:4q0
.
_user_specified_namem/conv_hid0_5x5/bias:6p2
0
_user_specified_namev/conv_hid0_5x5/kernel:6o2
0
_user_specified_namem/conv_hid0_5x5/kernel:4n0
.
_user_specified_namev/conv_hid0_4x4/bias:4m0
.
_user_specified_namem/conv_hid0_4x4/bias:6l2
0
_user_specified_namev/conv_hid0_4x4/kernel:6k2
0
_user_specified_namem/conv_hid0_4x4/kernel:4j0
.
_user_specified_namev/conv_hid0_3x3/bias:4i0
.
_user_specified_namem/conv_hid0_3x3/bias:6h2
0
_user_specified_namev/conv_hid0_3x3/kernel:6g2
0
_user_specified_namem/conv_hid0_3x3/kernel:5f1
/
_user_specified_namev/conv_prep1_5x5/bias:5e1
/
_user_specified_namem/conv_prep1_5x5/bias:7d3
1
_user_specified_namev/conv_prep1_5x5/kernel:7c3
1
_user_specified_namem/conv_prep1_5x5/kernel:5b1
/
_user_specified_namev/conv_prep1_4x4/bias:5a1
/
_user_specified_namem/conv_prep1_4x4/bias:7`3
1
_user_specified_namev/conv_prep1_4x4/kernel:7_3
1
_user_specified_namem/conv_prep1_4x4/kernel:5^1
/
_user_specified_namev/conv_prep1_3x3/bias:5]1
/
_user_specified_namem/conv_prep1_3x3/bias:7\3
1
_user_specified_namev/conv_prep1_3x3/kernel:7[3
1
_user_specified_namem/conv_prep1_3x3/kernel:5Z1
/
_user_specified_namev/conv_prep0_5x5/bias:5Y1
/
_user_specified_namem/conv_prep0_5x5/bias:7X3
1
_user_specified_namev/conv_prep0_5x5/kernel:7W3
1
_user_specified_namem/conv_prep0_5x5/kernel:5V1
/
_user_specified_namev/conv_prep0_4x4/bias:5U1
/
_user_specified_namem/conv_prep0_4x4/bias:7T3
1
_user_specified_namev/conv_prep0_4x4/kernel:7S3
1
_user_specified_namem/conv_prep0_4x4/kernel:5R1
/
_user_specified_namev/conv_prep0_3x3/bias:5Q1
/
_user_specified_namem/conv_prep0_3x3/bias:7P3
1
_user_specified_namev/conv_prep0_3x3/kernel:7O3
1
_user_specified_namem/conv_prep0_3x3/kernel:-N)
'
_user_specified_namelearning_rate:)M%
#
_user_specified_name	iteration:-L)
'
_user_specified_nameoutput_S/bias:/K+
)
_user_specified_nameoutput_S/kernel:2J.
,
_user_specified_nameconv_rev5_5x5/bias:4I0
.
_user_specified_nameconv_rev5_5x5/kernel:2H.
,
_user_specified_nameconv_rev4_4x4/bias:4G0
.
_user_specified_nameconv_rev4_4x4/kernel:2F.
,
_user_specified_nameconv_rev4_3x3/bias:4E0
.
_user_specified_nameconv_rev4_3x3/kernel:2D.
,
_user_specified_nameconv_rev3_5x5/bias:4C0
.
_user_specified_nameconv_rev3_5x5/kernel:2B.
,
_user_specified_nameconv_rev3_4x4/bias:4A0
.
_user_specified_nameconv_rev3_4x4/kernel:2@.
,
_user_specified_nameconv_rev3_3x3/bias:4?0
.
_user_specified_nameconv_rev3_3x3/kernel:2>.
,
_user_specified_nameconv_rev2_5x5/bias:4=0
.
_user_specified_nameconv_rev2_5x5/kernel:2<.
,
_user_specified_nameconv_rev2_4x4/bias:4;0
.
_user_specified_nameconv_rev2_4x4/kernel:2:.
,
_user_specified_nameconv_rev2_3x3/bias:490
.
_user_specified_nameconv_rev2_3x3/kernel:28.
,
_user_specified_nameconv_rev1_5x5/bias:470
.
_user_specified_nameconv_rev1_5x5/kernel:26.
,
_user_specified_nameconv_rev1_4x4/bias:450
.
_user_specified_nameconv_rev1_4x4/kernel:24.
,
_user_specified_nameconv_rev1_3x3/bias:430
.
_user_specified_nameconv_rev1_3x3/kernel:22.
,
_user_specified_nameconv_rev0_5x5/bias:410
.
_user_specified_nameconv_rev0_5x5/kernel:20.
,
_user_specified_nameconv_rev0_4x4/bias:4/0
.
_user_specified_nameconv_rev0_4x4/kernel:2..
,
_user_specified_nameconv_rev0_3x3/bias:4-0
.
_user_specified_nameconv_rev0_3x3/kernel:-,)
'
_user_specified_nameoutput_C/bias:/++
)
_user_specified_nameoutput_C/kernel:2*.
,
_user_specified_nameconv_hid5_5x5/bias:4)0
.
_user_specified_nameconv_hid5_5x5/kernel:2(.
,
_user_specified_nameconv_hid4_4x4/bias:4'0
.
_user_specified_nameconv_hid4_4x4/kernel:2&.
,
_user_specified_nameconv_hid4_3x3/bias:4%0
.
_user_specified_nameconv_hid4_3x3/kernel:2$.
,
_user_specified_nameconv_hid3_5x5/bias:4#0
.
_user_specified_nameconv_hid3_5x5/kernel:2".
,
_user_specified_nameconv_hid3_4x4/bias:4!0
.
_user_specified_nameconv_hid3_4x4/kernel:2 .
,
_user_specified_nameconv_hid3_3x3/bias:40
.
_user_specified_nameconv_hid3_3x3/kernel:2.
,
_user_specified_nameconv_hid2_5x5/bias:40
.
_user_specified_nameconv_hid2_5x5/kernel:2.
,
_user_specified_nameconv_hid2_4x4/bias:40
.
_user_specified_nameconv_hid2_4x4/kernel:2.
,
_user_specified_nameconv_hid2_3x3/bias:40
.
_user_specified_nameconv_hid2_3x3/kernel:2.
,
_user_specified_nameconv_hid1_5x5/bias:40
.
_user_specified_nameconv_hid1_5x5/kernel:2.
,
_user_specified_nameconv_hid1_4x4/bias:40
.
_user_specified_nameconv_hid1_4x4/kernel:2.
,
_user_specified_nameconv_hid1_3x3/bias:40
.
_user_specified_nameconv_hid1_3x3/kernel:2.
,
_user_specified_nameconv_hid0_5x5/bias:40
.
_user_specified_nameconv_hid0_5x5/kernel:2.
,
_user_specified_nameconv_hid0_4x4/bias:40
.
_user_specified_nameconv_hid0_4x4/kernel:2.
,
_user_specified_nameconv_hid0_3x3/bias:40
.
_user_specified_nameconv_hid0_3x3/kernel:3/
-
_user_specified_nameconv_prep1_5x5/bias:51
/
_user_specified_nameconv_prep1_5x5/kernel:3
/
-
_user_specified_nameconv_prep1_4x4/bias:5	1
/
_user_specified_nameconv_prep1_4x4/kernel:3/
-
_user_specified_nameconv_prep1_3x3/bias:51
/
_user_specified_nameconv_prep1_3x3/kernel:3/
-
_user_specified_nameconv_prep0_5x5/bias:51
/
_user_specified_nameconv_prep0_5x5/kernel:3/
-
_user_specified_nameconv_prep0_4x4/bias:51
/
_user_specified_nameconv_prep0_4x4/kernel:3/
-
_user_specified_nameconv_prep0_3x3/bias:51
/
_user_specified_nameconv_prep0_3x3/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
,__inference_conv_rev2_3x3_layer_call_fn_5566

inputs!
unknown:A2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev2_3x3_layer_call_and_return_conditional_losses_3402w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@2<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5562:$ 

_user_specified_name5560:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
,__inference_conv_rev3_3x3_layer_call_fn_5641

inputs!
unknown:A2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev3_3x3_layer_call_and_return_conditional_losses_3459w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@2<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5637:$ 

_user_specified_name5635:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
X
,__inference_concatenate_2_layer_call_fn_4980
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@D* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_2_layer_call_and_return_conditional_losses_2395h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@D"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������@@:���������@@A:YU
/
_output_shapes
:���������@@A
"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_0
�
�
B__inference_output_S_layer_call_and_return_conditional_losses_5802

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�E
�
?__inference_model_layer_call_and_return_conditional_losses_4328
input_1
input_2&
encoder_4172:2
encoder_4174:2&
encoder_4176:

encoder_4178:
&
encoder_4180:
encoder_4182:&
encoder_4184:A2
encoder_4186:2&
encoder_4188:A

encoder_4190:
&
encoder_4192:A
encoder_4194:&
encoder_4196:D2
encoder_4198:2&
encoder_4200:D

encoder_4202:
&
encoder_4204:D
encoder_4206:&
encoder_4208:A2
encoder_4210:2&
encoder_4212:A

encoder_4214:
&
encoder_4216:A
encoder_4218:&
encoder_4220:A2
encoder_4222:2&
encoder_4224:A

encoder_4226:
&
encoder_4228:A
encoder_4230:&
encoder_4232:A2
encoder_4234:2&
encoder_4236:A

encoder_4238:
&
encoder_4240:A
encoder_4242:&
encoder_4244:A2
encoder_4246:2&
encoder_4248:A

encoder_4250:
&
encoder_4252:A
encoder_4254:&
encoder_4256:A
encoder_4258:&
decoder_4261:2
decoder_4263:2&
decoder_4265:

decoder_4267:
&
decoder_4269:
decoder_4271:&
decoder_4273:A2
decoder_4275:2&
decoder_4277:A

decoder_4279:
&
decoder_4281:A
decoder_4283:&
decoder_4285:A2
decoder_4287:2&
decoder_4289:A

decoder_4291:
&
decoder_4293:A
decoder_4295:&
decoder_4297:A2
decoder_4299:2&
decoder_4301:A

decoder_4303:
&
decoder_4305:A
decoder_4307:&
decoder_4309:A2
decoder_4311:2&
decoder_4313:A

decoder_4315:
&
decoder_4317:A
decoder_4319:&
decoder_4321:A
decoder_4323:
identity��Decoder/StatefulPartitionedCall�Encoder/StatefulPartitionedCall�
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2encoder_4172encoder_4174encoder_4176encoder_4178encoder_4180encoder_4182encoder_4184encoder_4186encoder_4188encoder_4190encoder_4192encoder_4194encoder_4196encoder_4198encoder_4200encoder_4202encoder_4204encoder_4206encoder_4208encoder_4210encoder_4212encoder_4214encoder_4216encoder_4218encoder_4220encoder_4222encoder_4224encoder_4226encoder_4228encoder_4230encoder_4232encoder_4234encoder_4236encoder_4238encoder_4240encoder_4242encoder_4244encoder_4246encoder_4248encoder_4250encoder_4252encoder_4254encoder_4256encoder_4258*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,-*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_Encoder_layer_call_and_return_conditional_losses_2822�
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_4261decoder_4263decoder_4265decoder_4267decoder_4269decoder_4271decoder_4273decoder_4275decoder_4277decoder_4279decoder_4281decoder_4283decoder_4285decoder_4287decoder_4289decoder_4291decoder_4293decoder_4295decoder_4297decoder_4299decoder_4301decoder_4303decoder_4305decoder_4307decoder_4309decoder_4311decoder_4313decoder_4315decoder_4317decoder_4319decoder_4321decoder_4323*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_Decoder_layer_call_and_return_conditional_losses_3674�
concatenate_13/PartitionedCallPartitionedCall(Decoder/StatefulPartitionedCall:output:0(Encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_concatenate_13_layer_call_and_return_conditional_losses_4165~
IdentityIdentity'concatenate_13/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@f
NoOpNoOp ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@@:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:$M 

_user_specified_name4323:$L 

_user_specified_name4321:$K 

_user_specified_name4319:$J 

_user_specified_name4317:$I 

_user_specified_name4315:$H 

_user_specified_name4313:$G 

_user_specified_name4311:$F 

_user_specified_name4309:$E 

_user_specified_name4307:$D 

_user_specified_name4305:$C 

_user_specified_name4303:$B 

_user_specified_name4301:$A 

_user_specified_name4299:$@ 

_user_specified_name4297:$? 

_user_specified_name4295:$> 

_user_specified_name4293:$= 

_user_specified_name4291:$< 

_user_specified_name4289:$; 

_user_specified_name4287:$: 

_user_specified_name4285:$9 

_user_specified_name4283:$8 

_user_specified_name4281:$7 

_user_specified_name4279:$6 

_user_specified_name4277:$5 

_user_specified_name4275:$4 

_user_specified_name4273:$3 

_user_specified_name4271:$2 

_user_specified_name4269:$1 

_user_specified_name4267:$0 

_user_specified_name4265:$/ 

_user_specified_name4263:$. 

_user_specified_name4261:$- 

_user_specified_name4258:$, 

_user_specified_name4256:$+ 

_user_specified_name4254:$* 

_user_specified_name4252:$) 

_user_specified_name4250:$( 

_user_specified_name4248:$' 

_user_specified_name4246:$& 

_user_specified_name4244:$% 

_user_specified_name4242:$$ 

_user_specified_name4240:$# 

_user_specified_name4238:$" 

_user_specified_name4236:$! 

_user_specified_name4234:$  

_user_specified_name4232:$ 

_user_specified_name4230:$ 

_user_specified_name4228:$ 

_user_specified_name4226:$ 

_user_specified_name4224:$ 

_user_specified_name4222:$ 

_user_specified_name4220:$ 

_user_specified_name4218:$ 

_user_specified_name4216:$ 

_user_specified_name4214:$ 

_user_specified_name4212:$ 

_user_specified_name4210:$ 

_user_specified_name4208:$ 

_user_specified_name4206:$ 

_user_specified_name4204:$ 

_user_specified_name4202:$ 

_user_specified_name4200:$ 

_user_specified_name4198:$ 

_user_specified_name4196:$ 

_user_specified_name4194:$ 

_user_specified_name4192:$ 

_user_specified_name4190:$
 

_user_specified_name4188:$	 

_user_specified_name4186:$ 

_user_specified_name4184:$ 

_user_specified_name4182:$ 

_user_specified_name4180:$ 

_user_specified_name4178:$ 

_user_specified_name4176:$ 

_user_specified_name4174:$ 

_user_specified_name4172:XT
/
_output_shapes
:���������@@
!
_user_specified_name	input_2:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_1
�
�
H__inference_conv_prep1_4x4_layer_call_and_return_conditional_losses_2358

inputs8
conv2d_readvariableop_resource:A
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_rev3_3x3_layer_call_and_return_conditional_losses_5652

inputs8
conv2d_readvariableop_resource:A2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
,__inference_conv_rev4_4x4_layer_call_fn_5736

inputs!
unknown:A

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev4_4x4_layer_call_and_return_conditional_losses_3532w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5732:$ 

_user_specified_name5730:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
B__inference_output_C_layer_call_and_return_conditional_losses_5382

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_hid2_5x5_layer_call_and_return_conditional_losses_5197

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�

G__inference_concatenate_8_layer_call_and_return_conditional_losses_3333

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:WS
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������@@

 
_user_specified_nameinputs:W S
/
_output_shapes
:���������@@2
 
_user_specified_nameinputs
�:
�
$__inference_model_layer_call_fn_4486
input_1
input_2!
unknown:2
	unknown_0:2#
	unknown_1:

	unknown_2:
#
	unknown_3:
	unknown_4:#
	unknown_5:A2
	unknown_6:2#
	unknown_7:A

	unknown_8:
#
	unknown_9:A

unknown_10:$

unknown_11:D2

unknown_12:2$

unknown_13:D


unknown_14:
$

unknown_15:D

unknown_16:$

unknown_17:A2

unknown_18:2$

unknown_19:A


unknown_20:
$

unknown_21:A

unknown_22:$

unknown_23:A2

unknown_24:2$

unknown_25:A


unknown_26:
$

unknown_27:A

unknown_28:$

unknown_29:A2

unknown_30:2$

unknown_31:A


unknown_32:
$

unknown_33:A

unknown_34:$

unknown_35:A2

unknown_36:2$

unknown_37:A


unknown_38:
$

unknown_39:A

unknown_40:$

unknown_41:A

unknown_42:$

unknown_43:2

unknown_44:2$

unknown_45:


unknown_46:
$

unknown_47:

unknown_48:$

unknown_49:A2

unknown_50:2$

unknown_51:A


unknown_52:
$

unknown_53:A

unknown_54:$

unknown_55:A2

unknown_56:2$

unknown_57:A


unknown_58:
$

unknown_59:A

unknown_60:$

unknown_61:A2

unknown_62:2$

unknown_63:A


unknown_64:
$

unknown_65:A

unknown_66:$

unknown_67:A2

unknown_68:2$

unknown_69:A


unknown_70:
$

unknown_71:A

unknown_72:$

unknown_73:A

unknown_74:
identity��StatefulPartitionedCall�

StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74*Y
TinR
P2N*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*n
_read_only_resource_inputsP
NL	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLM*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_4168w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@@:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$M 

_user_specified_name4482:$L 

_user_specified_name4480:$K 

_user_specified_name4478:$J 

_user_specified_name4476:$I 

_user_specified_name4474:$H 

_user_specified_name4472:$G 

_user_specified_name4470:$F 

_user_specified_name4468:$E 

_user_specified_name4466:$D 

_user_specified_name4464:$C 

_user_specified_name4462:$B 

_user_specified_name4460:$A 

_user_specified_name4458:$@ 

_user_specified_name4456:$? 

_user_specified_name4454:$> 

_user_specified_name4452:$= 

_user_specified_name4450:$< 

_user_specified_name4448:$; 

_user_specified_name4446:$: 

_user_specified_name4444:$9 

_user_specified_name4442:$8 

_user_specified_name4440:$7 

_user_specified_name4438:$6 

_user_specified_name4436:$5 

_user_specified_name4434:$4 

_user_specified_name4432:$3 

_user_specified_name4430:$2 

_user_specified_name4428:$1 

_user_specified_name4426:$0 

_user_specified_name4424:$/ 

_user_specified_name4422:$. 

_user_specified_name4420:$- 

_user_specified_name4418:$, 

_user_specified_name4416:$+ 

_user_specified_name4414:$* 

_user_specified_name4412:$) 

_user_specified_name4410:$( 

_user_specified_name4408:$' 

_user_specified_name4406:$& 

_user_specified_name4404:$% 

_user_specified_name4402:$$ 

_user_specified_name4400:$# 

_user_specified_name4398:$" 

_user_specified_name4396:$! 

_user_specified_name4394:$  

_user_specified_name4392:$ 

_user_specified_name4390:$ 

_user_specified_name4388:$ 

_user_specified_name4386:$ 

_user_specified_name4384:$ 

_user_specified_name4382:$ 

_user_specified_name4380:$ 

_user_specified_name4378:$ 

_user_specified_name4376:$ 

_user_specified_name4374:$ 

_user_specified_name4372:$ 

_user_specified_name4370:$ 

_user_specified_name4368:$ 

_user_specified_name4366:$ 

_user_specified_name4364:$ 

_user_specified_name4362:$ 

_user_specified_name4360:$ 

_user_specified_name4358:$ 

_user_specified_name4356:$ 

_user_specified_name4354:$ 

_user_specified_name4352:$ 

_user_specified_name4350:$
 

_user_specified_name4348:$	 

_user_specified_name4346:$ 

_user_specified_name4344:$ 

_user_specified_name4342:$ 

_user_specified_name4340:$ 

_user_specified_name4338:$ 

_user_specified_name4336:$ 

_user_specified_name4334:$ 

_user_specified_name4332:XT
/
_output_shapes
:���������@@
!
_user_specified_name	input_2:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_1
�

G__inference_concatenate_9_layer_call_and_return_conditional_losses_3390

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:WS
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������@@

 
_user_specified_nameinputs:W S
/
_output_shapes
:���������@@2
 
_user_specified_nameinputs
�
�
H__inference_conv_prep1_3x3_layer_call_and_return_conditional_losses_4919

inputs8
conv2d_readvariableop_resource:A2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�:
�
"__inference_signature_wrapper_4811
input_1
input_2!
unknown:2
	unknown_0:2#
	unknown_1:

	unknown_2:
#
	unknown_3:
	unknown_4:#
	unknown_5:A2
	unknown_6:2#
	unknown_7:A

	unknown_8:
#
	unknown_9:A

unknown_10:$

unknown_11:D2

unknown_12:2$

unknown_13:D


unknown_14:
$

unknown_15:D

unknown_16:$

unknown_17:A2

unknown_18:2$

unknown_19:A


unknown_20:
$

unknown_21:A

unknown_22:$

unknown_23:A2

unknown_24:2$

unknown_25:A


unknown_26:
$

unknown_27:A

unknown_28:$

unknown_29:A2

unknown_30:2$

unknown_31:A


unknown_32:
$

unknown_33:A

unknown_34:$

unknown_35:A2

unknown_36:2$

unknown_37:A


unknown_38:
$

unknown_39:A

unknown_40:$

unknown_41:A

unknown_42:$

unknown_43:2

unknown_44:2$

unknown_45:


unknown_46:
$

unknown_47:

unknown_48:$

unknown_49:A2

unknown_50:2$

unknown_51:A


unknown_52:
$

unknown_53:A

unknown_54:$

unknown_55:A2

unknown_56:2$

unknown_57:A


unknown_58:
$

unknown_59:A

unknown_60:$

unknown_61:A2

unknown_62:2$

unknown_63:A


unknown_64:
$

unknown_65:A

unknown_66:$

unknown_67:A2

unknown_68:2$

unknown_69:A


unknown_70:
$

unknown_71:A

unknown_72:$

unknown_73:A

unknown_74:
identity��StatefulPartitionedCall�

StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74*Y
TinR
P2N*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*n
_read_only_resource_inputsP
NL	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLM*-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__wrapped_model_2271w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@@:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$M 

_user_specified_name4807:$L 

_user_specified_name4805:$K 

_user_specified_name4803:$J 

_user_specified_name4801:$I 

_user_specified_name4799:$H 

_user_specified_name4797:$G 

_user_specified_name4795:$F 

_user_specified_name4793:$E 

_user_specified_name4791:$D 

_user_specified_name4789:$C 

_user_specified_name4787:$B 

_user_specified_name4785:$A 

_user_specified_name4783:$@ 

_user_specified_name4781:$? 

_user_specified_name4779:$> 

_user_specified_name4777:$= 

_user_specified_name4775:$< 

_user_specified_name4773:$; 

_user_specified_name4771:$: 

_user_specified_name4769:$9 

_user_specified_name4767:$8 

_user_specified_name4765:$7 

_user_specified_name4763:$6 

_user_specified_name4761:$5 

_user_specified_name4759:$4 

_user_specified_name4757:$3 

_user_specified_name4755:$2 

_user_specified_name4753:$1 

_user_specified_name4751:$0 

_user_specified_name4749:$/ 

_user_specified_name4747:$. 

_user_specified_name4745:$- 

_user_specified_name4743:$, 

_user_specified_name4741:$+ 

_user_specified_name4739:$* 

_user_specified_name4737:$) 

_user_specified_name4735:$( 

_user_specified_name4733:$' 

_user_specified_name4731:$& 

_user_specified_name4729:$% 

_user_specified_name4727:$$ 

_user_specified_name4725:$# 

_user_specified_name4723:$" 

_user_specified_name4721:$! 

_user_specified_name4719:$  

_user_specified_name4717:$ 

_user_specified_name4715:$ 

_user_specified_name4713:$ 

_user_specified_name4711:$ 

_user_specified_name4709:$ 

_user_specified_name4707:$ 

_user_specified_name4705:$ 

_user_specified_name4703:$ 

_user_specified_name4701:$ 

_user_specified_name4699:$ 

_user_specified_name4697:$ 

_user_specified_name4695:$ 

_user_specified_name4693:$ 

_user_specified_name4691:$ 

_user_specified_name4689:$ 

_user_specified_name4687:$ 

_user_specified_name4685:$ 

_user_specified_name4683:$ 

_user_specified_name4681:$ 

_user_specified_name4679:$ 

_user_specified_name4677:$ 

_user_specified_name4675:$
 

_user_specified_name4673:$	 

_user_specified_name4671:$ 

_user_specified_name4669:$ 

_user_specified_name4667:$ 

_user_specified_name4665:$ 

_user_specified_name4663:$ 

_user_specified_name4661:$ 

_user_specified_name4659:$ 

_user_specified_name4657:XT
/
_output_shapes
:���������@@
!
_user_specified_name	input_2:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_1
�
�
G__inference_conv_rev3_3x3_layer_call_and_return_conditional_losses_3459

inputs8
conv2d_readvariableop_resource:A2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
H__inference_conv_prep1_4x4_layer_call_and_return_conditional_losses_4939

inputs8
conv2d_readvariableop_resource:A
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�u
�
A__inference_Decoder_layer_call_and_return_conditional_losses_3674
input_5,
conv_rev0_3x3_3588:2 
conv_rev0_3x3_3590:2,
conv_rev0_4x4_3593:
 
conv_rev0_4x4_3595:
,
conv_rev0_5x5_3598: 
conv_rev0_5x5_3600:,
conv_rev1_3x3_3604:A2 
conv_rev1_3x3_3606:2,
conv_rev1_4x4_3609:A
 
conv_rev1_4x4_3611:
,
conv_rev1_5x5_3614:A 
conv_rev1_5x5_3616:,
conv_rev2_3x3_3620:A2 
conv_rev2_3x3_3622:2,
conv_rev2_4x4_3625:A
 
conv_rev2_4x4_3627:
,
conv_rev2_5x5_3630:A 
conv_rev2_5x5_3632:,
conv_rev3_3x3_3636:A2 
conv_rev3_3x3_3638:2,
conv_rev3_4x4_3641:A
 
conv_rev3_4x4_3643:
,
conv_rev3_5x5_3646:A 
conv_rev3_5x5_3648:,
conv_rev4_3x3_3652:A2 
conv_rev4_3x3_3654:2,
conv_rev4_4x4_3657:A
 
conv_rev4_4x4_3659:
,
conv_rev5_5x5_3662:A 
conv_rev5_5x5_3664:'
output_s_3668:A
output_s_3670:
identity��%conv_rev0_3x3/StatefulPartitionedCall�%conv_rev0_4x4/StatefulPartitionedCall�%conv_rev0_5x5/StatefulPartitionedCall�%conv_rev1_3x3/StatefulPartitionedCall�%conv_rev1_4x4/StatefulPartitionedCall�%conv_rev1_5x5/StatefulPartitionedCall�%conv_rev2_3x3/StatefulPartitionedCall�%conv_rev2_4x4/StatefulPartitionedCall�%conv_rev2_5x5/StatefulPartitionedCall�%conv_rev3_3x3/StatefulPartitionedCall�%conv_rev3_4x4/StatefulPartitionedCall�%conv_rev3_5x5/StatefulPartitionedCall�%conv_rev4_3x3/StatefulPartitionedCall�%conv_rev4_4x4/StatefulPartitionedCall�%conv_rev5_5x5/StatefulPartitionedCall� output_S/StatefulPartitionedCall�
output_C_noise/PartitionedCallPartitionedCallinput_5*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_output_C_noise_layer_call_and_return_conditional_losses_3586�
%conv_rev0_3x3/StatefulPartitionedCallStatefulPartitionedCall'output_C_noise/PartitionedCall:output:0conv_rev0_3x3_3588conv_rev0_3x3_3590*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev0_3x3_layer_call_and_return_conditional_losses_3288�
%conv_rev0_4x4/StatefulPartitionedCallStatefulPartitionedCall'output_C_noise/PartitionedCall:output:0conv_rev0_4x4_3593conv_rev0_4x4_3595*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev0_4x4_layer_call_and_return_conditional_losses_3304�
%conv_rev0_5x5/StatefulPartitionedCallStatefulPartitionedCall'output_C_noise/PartitionedCall:output:0conv_rev0_5x5_3598conv_rev0_5x5_3600*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev0_5x5_layer_call_and_return_conditional_losses_3320�
concatenate_8/PartitionedCallPartitionedCall.conv_rev0_3x3/StatefulPartitionedCall:output:0.conv_rev0_4x4/StatefulPartitionedCall:output:0.conv_rev0_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_8_layer_call_and_return_conditional_losses_3333�
%conv_rev1_3x3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0conv_rev1_3x3_3604conv_rev1_3x3_3606*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev1_3x3_layer_call_and_return_conditional_losses_3345�
%conv_rev1_4x4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0conv_rev1_4x4_3609conv_rev1_4x4_3611*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev1_4x4_layer_call_and_return_conditional_losses_3361�
%conv_rev1_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0conv_rev1_5x5_3614conv_rev1_5x5_3616*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev1_5x5_layer_call_and_return_conditional_losses_3377�
concatenate_9/PartitionedCallPartitionedCall.conv_rev1_3x3/StatefulPartitionedCall:output:0.conv_rev1_4x4/StatefulPartitionedCall:output:0.conv_rev1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_9_layer_call_and_return_conditional_losses_3390�
%conv_rev2_3x3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0conv_rev2_3x3_3620conv_rev2_3x3_3622*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev2_3x3_layer_call_and_return_conditional_losses_3402�
%conv_rev2_4x4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0conv_rev2_4x4_3625conv_rev2_4x4_3627*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev2_4x4_layer_call_and_return_conditional_losses_3418�
%conv_rev2_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0conv_rev2_5x5_3630conv_rev2_5x5_3632*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev2_5x5_layer_call_and_return_conditional_losses_3434�
concatenate_10/PartitionedCallPartitionedCall.conv_rev2_3x3/StatefulPartitionedCall:output:0.conv_rev2_4x4/StatefulPartitionedCall:output:0.conv_rev2_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_concatenate_10_layer_call_and_return_conditional_losses_3447�
%conv_rev3_3x3/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0conv_rev3_3x3_3636conv_rev3_3x3_3638*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev3_3x3_layer_call_and_return_conditional_losses_3459�
%conv_rev3_4x4/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0conv_rev3_4x4_3641conv_rev3_4x4_3643*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev3_4x4_layer_call_and_return_conditional_losses_3475�
%conv_rev3_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0conv_rev3_5x5_3646conv_rev3_5x5_3648*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev3_5x5_layer_call_and_return_conditional_losses_3491�
concatenate_11/PartitionedCallPartitionedCall.conv_rev3_3x3/StatefulPartitionedCall:output:0.conv_rev3_4x4/StatefulPartitionedCall:output:0.conv_rev3_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_concatenate_11_layer_call_and_return_conditional_losses_3504�
%conv_rev4_3x3/StatefulPartitionedCallStatefulPartitionedCall'concatenate_11/PartitionedCall:output:0conv_rev4_3x3_3652conv_rev4_3x3_3654*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev4_3x3_layer_call_and_return_conditional_losses_3516�
%conv_rev4_4x4/StatefulPartitionedCallStatefulPartitionedCall'concatenate_11/PartitionedCall:output:0conv_rev4_4x4_3657conv_rev4_4x4_3659*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev4_4x4_layer_call_and_return_conditional_losses_3532�
%conv_rev5_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_11/PartitionedCall:output:0conv_rev5_5x5_3662conv_rev5_5x5_3664*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev5_5x5_layer_call_and_return_conditional_losses_3548�
concatenate_12/PartitionedCallPartitionedCall.conv_rev4_3x3/StatefulPartitionedCall:output:0.conv_rev4_4x4/StatefulPartitionedCall:output:0.conv_rev5_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_concatenate_12_layer_call_and_return_conditional_losses_3561�
 output_S/StatefulPartitionedCallStatefulPartitionedCall'concatenate_12/PartitionedCall:output:0output_s_3668output_s_3670*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_output_S_layer_call_and_return_conditional_losses_3573�
IdentityIdentity)output_S/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@�
NoOpNoOp&^conv_rev0_3x3/StatefulPartitionedCall&^conv_rev0_4x4/StatefulPartitionedCall&^conv_rev0_5x5/StatefulPartitionedCall&^conv_rev1_3x3/StatefulPartitionedCall&^conv_rev1_4x4/StatefulPartitionedCall&^conv_rev1_5x5/StatefulPartitionedCall&^conv_rev2_3x3/StatefulPartitionedCall&^conv_rev2_4x4/StatefulPartitionedCall&^conv_rev2_5x5/StatefulPartitionedCall&^conv_rev3_3x3/StatefulPartitionedCall&^conv_rev3_4x4/StatefulPartitionedCall&^conv_rev3_5x5/StatefulPartitionedCall&^conv_rev4_3x3/StatefulPartitionedCall&^conv_rev4_4x4/StatefulPartitionedCall&^conv_rev5_5x5/StatefulPartitionedCall!^output_S/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%conv_rev0_3x3/StatefulPartitionedCall%conv_rev0_3x3/StatefulPartitionedCall2N
%conv_rev0_4x4/StatefulPartitionedCall%conv_rev0_4x4/StatefulPartitionedCall2N
%conv_rev0_5x5/StatefulPartitionedCall%conv_rev0_5x5/StatefulPartitionedCall2N
%conv_rev1_3x3/StatefulPartitionedCall%conv_rev1_3x3/StatefulPartitionedCall2N
%conv_rev1_4x4/StatefulPartitionedCall%conv_rev1_4x4/StatefulPartitionedCall2N
%conv_rev1_5x5/StatefulPartitionedCall%conv_rev1_5x5/StatefulPartitionedCall2N
%conv_rev2_3x3/StatefulPartitionedCall%conv_rev2_3x3/StatefulPartitionedCall2N
%conv_rev2_4x4/StatefulPartitionedCall%conv_rev2_4x4/StatefulPartitionedCall2N
%conv_rev2_5x5/StatefulPartitionedCall%conv_rev2_5x5/StatefulPartitionedCall2N
%conv_rev3_3x3/StatefulPartitionedCall%conv_rev3_3x3/StatefulPartitionedCall2N
%conv_rev3_4x4/StatefulPartitionedCall%conv_rev3_4x4/StatefulPartitionedCall2N
%conv_rev3_5x5/StatefulPartitionedCall%conv_rev3_5x5/StatefulPartitionedCall2N
%conv_rev4_3x3/StatefulPartitionedCall%conv_rev4_3x3/StatefulPartitionedCall2N
%conv_rev4_4x4/StatefulPartitionedCall%conv_rev4_4x4/StatefulPartitionedCall2N
%conv_rev5_5x5/StatefulPartitionedCall%conv_rev5_5x5/StatefulPartitionedCall2D
 output_S/StatefulPartitionedCall output_S/StatefulPartitionedCall:$  

_user_specified_name3670:$ 

_user_specified_name3668:$ 

_user_specified_name3664:$ 

_user_specified_name3662:$ 

_user_specified_name3659:$ 

_user_specified_name3657:$ 

_user_specified_name3654:$ 

_user_specified_name3652:$ 

_user_specified_name3648:$ 

_user_specified_name3646:$ 

_user_specified_name3643:$ 

_user_specified_name3641:$ 

_user_specified_name3638:$ 

_user_specified_name3636:$ 

_user_specified_name3632:$ 

_user_specified_name3630:$ 

_user_specified_name3627:$ 

_user_specified_name3625:$ 

_user_specified_name3622:$ 

_user_specified_name3620:$ 

_user_specified_name3616:$ 

_user_specified_name3614:$
 

_user_specified_name3611:$	 

_user_specified_name3609:$ 

_user_specified_name3606:$ 

_user_specified_name3604:$ 

_user_specified_name3600:$ 

_user_specified_name3598:$ 

_user_specified_name3595:$ 

_user_specified_name3593:$ 

_user_specified_name3590:$ 

_user_specified_name3588:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_5
�
�
G__inference_conv_rev4_3x3_layer_call_and_return_conditional_losses_5727

inputs8
conv2d_readvariableop_resource:A2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
H__inference_concatenate_12_layer_call_and_return_conditional_losses_5782
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
�
-__inference_conv_prep0_5x5_layer_call_fn_4873

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv_prep0_5x5_layer_call_and_return_conditional_losses_2317w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name4869:$ 

_user_specified_name4867:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
G__inference_conv_rev2_5x5_layer_call_and_return_conditional_losses_5617

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�E
�
?__inference_model_layer_call_and_return_conditional_losses_4168
input_1
input_2&
encoder_4005:2
encoder_4007:2&
encoder_4009:

encoder_4011:
&
encoder_4013:
encoder_4015:&
encoder_4017:A2
encoder_4019:2&
encoder_4021:A

encoder_4023:
&
encoder_4025:A
encoder_4027:&
encoder_4029:D2
encoder_4031:2&
encoder_4033:D

encoder_4035:
&
encoder_4037:D
encoder_4039:&
encoder_4041:A2
encoder_4043:2&
encoder_4045:A

encoder_4047:
&
encoder_4049:A
encoder_4051:&
encoder_4053:A2
encoder_4055:2&
encoder_4057:A

encoder_4059:
&
encoder_4061:A
encoder_4063:&
encoder_4065:A2
encoder_4067:2&
encoder_4069:A

encoder_4071:
&
encoder_4073:A
encoder_4075:&
encoder_4077:A2
encoder_4079:2&
encoder_4081:A

encoder_4083:
&
encoder_4085:A
encoder_4087:&
encoder_4089:A
encoder_4091:&
decoder_4094:2
decoder_4096:2&
decoder_4098:

decoder_4100:
&
decoder_4102:
decoder_4104:&
decoder_4106:A2
decoder_4108:2&
decoder_4110:A

decoder_4112:
&
decoder_4114:A
decoder_4116:&
decoder_4118:A2
decoder_4120:2&
decoder_4122:A

decoder_4124:
&
decoder_4126:A
decoder_4128:&
decoder_4130:A2
decoder_4132:2&
decoder_4134:A

decoder_4136:
&
decoder_4138:A
decoder_4140:&
decoder_4142:A2
decoder_4144:2&
decoder_4146:A

decoder_4148:
&
decoder_4150:A
decoder_4152:&
decoder_4154:A
decoder_4156:
identity��Decoder/StatefulPartitionedCall�Encoder/StatefulPartitionedCall�
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2encoder_4005encoder_4007encoder_4009encoder_4011encoder_4013encoder_4015encoder_4017encoder_4019encoder_4021encoder_4023encoder_4025encoder_4027encoder_4029encoder_4031encoder_4033encoder_4035encoder_4037encoder_4039encoder_4041encoder_4043encoder_4045encoder_4047encoder_4049encoder_4051encoder_4053encoder_4055encoder_4057encoder_4059encoder_4061encoder_4063encoder_4065encoder_4067encoder_4069encoder_4071encoder_4073encoder_4075encoder_4077encoder_4079encoder_4081encoder_4083encoder_4085encoder_4087encoder_4089encoder_4091*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,-*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_Encoder_layer_call_and_return_conditional_losses_2699�
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_4094decoder_4096decoder_4098decoder_4100decoder_4102decoder_4104decoder_4106decoder_4108decoder_4110decoder_4112decoder_4114decoder_4116decoder_4118decoder_4120decoder_4122decoder_4124decoder_4126decoder_4128decoder_4130decoder_4132decoder_4134decoder_4136decoder_4138decoder_4140decoder_4142decoder_4144decoder_4146decoder_4148decoder_4150decoder_4152decoder_4154decoder_4156*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_Decoder_layer_call_and_return_conditional_losses_3580�
concatenate_13/PartitionedCallPartitionedCall(Decoder/StatefulPartitionedCall:output:0(Encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_concatenate_13_layer_call_and_return_conditional_losses_4165~
IdentityIdentity'concatenate_13/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@f
NoOpNoOp ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@@:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:$M 

_user_specified_name4156:$L 

_user_specified_name4154:$K 

_user_specified_name4152:$J 

_user_specified_name4150:$I 

_user_specified_name4148:$H 

_user_specified_name4146:$G 

_user_specified_name4144:$F 

_user_specified_name4142:$E 

_user_specified_name4140:$D 

_user_specified_name4138:$C 

_user_specified_name4136:$B 

_user_specified_name4134:$A 

_user_specified_name4132:$@ 

_user_specified_name4130:$? 

_user_specified_name4128:$> 

_user_specified_name4126:$= 

_user_specified_name4124:$< 

_user_specified_name4122:$; 

_user_specified_name4120:$: 

_user_specified_name4118:$9 

_user_specified_name4116:$8 

_user_specified_name4114:$7 

_user_specified_name4112:$6 

_user_specified_name4110:$5 

_user_specified_name4108:$4 

_user_specified_name4106:$3 

_user_specified_name4104:$2 

_user_specified_name4102:$1 

_user_specified_name4100:$0 

_user_specified_name4098:$/ 

_user_specified_name4096:$. 

_user_specified_name4094:$- 

_user_specified_name4091:$, 

_user_specified_name4089:$+ 

_user_specified_name4087:$* 

_user_specified_name4085:$) 

_user_specified_name4083:$( 

_user_specified_name4081:$' 

_user_specified_name4079:$& 

_user_specified_name4077:$% 

_user_specified_name4075:$$ 

_user_specified_name4073:$# 

_user_specified_name4071:$" 

_user_specified_name4069:$! 

_user_specified_name4067:$  

_user_specified_name4065:$ 

_user_specified_name4063:$ 

_user_specified_name4061:$ 

_user_specified_name4059:$ 

_user_specified_name4057:$ 

_user_specified_name4055:$ 

_user_specified_name4053:$ 

_user_specified_name4051:$ 

_user_specified_name4049:$ 

_user_specified_name4047:$ 

_user_specified_name4045:$ 

_user_specified_name4043:$ 

_user_specified_name4041:$ 

_user_specified_name4039:$ 

_user_specified_name4037:$ 

_user_specified_name4035:$ 

_user_specified_name4033:$ 

_user_specified_name4031:$ 

_user_specified_name4029:$ 

_user_specified_name4027:$ 

_user_specified_name4025:$ 

_user_specified_name4023:$
 

_user_specified_name4021:$	 

_user_specified_name4019:$ 

_user_specified_name4017:$ 

_user_specified_name4015:$ 

_user_specified_name4013:$ 

_user_specified_name4011:$ 

_user_specified_name4009:$ 

_user_specified_name4007:$ 

_user_specified_name4005:XT
/
_output_shapes
:���������@@
!
_user_specified_name	input_2:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_1
�
�
G__inference_conv_rev2_4x4_layer_call_and_return_conditional_losses_5597

inputs8
conv2d_readvariableop_resource:A
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_hid3_3x3_layer_call_and_return_conditional_losses_5232

inputs8
conv2d_readvariableop_resource:A2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�

E__inference_concatenate_layer_call_and_return_conditional_losses_4899
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:YU
/
_output_shapes
:���������@@
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:���������@@

"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������@@2
"
_user_specified_name
inputs_0
�
�
,__inference_conv_rev3_5x5_layer_call_fn_5681

inputs!
unknown:A
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev3_5x5_layer_call_and_return_conditional_losses_3491w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5677:$ 

_user_specified_name5675:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�%
�
&__inference_Encoder_layer_call_fn_2916
input_3
input_4!
unknown:2
	unknown_0:2#
	unknown_1:

	unknown_2:
#
	unknown_3:
	unknown_4:#
	unknown_5:A2
	unknown_6:2#
	unknown_7:A

	unknown_8:
#
	unknown_9:A

unknown_10:$

unknown_11:D2

unknown_12:2$

unknown_13:D


unknown_14:
$

unknown_15:D

unknown_16:$

unknown_17:A2

unknown_18:2$

unknown_19:A


unknown_20:
$

unknown_21:A

unknown_22:$

unknown_23:A2

unknown_24:2$

unknown_25:A


unknown_26:
$

unknown_27:A

unknown_28:$

unknown_29:A2

unknown_30:2$

unknown_31:A


unknown_32:
$

unknown_33:A

unknown_34:$

unknown_35:A2

unknown_36:2$

unknown_37:A


unknown_38:
$

unknown_39:A

unknown_40:$

unknown_41:A

unknown_42:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,-*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_Encoder_layer_call_and_return_conditional_losses_2699w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@@:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$- 

_user_specified_name2912:$, 

_user_specified_name2910:$+ 

_user_specified_name2908:$* 

_user_specified_name2906:$) 

_user_specified_name2904:$( 

_user_specified_name2902:$' 

_user_specified_name2900:$& 

_user_specified_name2898:$% 

_user_specified_name2896:$$ 

_user_specified_name2894:$# 

_user_specified_name2892:$" 

_user_specified_name2890:$! 

_user_specified_name2888:$  

_user_specified_name2886:$ 

_user_specified_name2884:$ 

_user_specified_name2882:$ 

_user_specified_name2880:$ 

_user_specified_name2878:$ 

_user_specified_name2876:$ 

_user_specified_name2874:$ 

_user_specified_name2872:$ 

_user_specified_name2870:$ 

_user_specified_name2868:$ 

_user_specified_name2866:$ 

_user_specified_name2864:$ 

_user_specified_name2862:$ 

_user_specified_name2860:$ 

_user_specified_name2858:$ 

_user_specified_name2856:$ 

_user_specified_name2854:$ 

_user_specified_name2852:$ 

_user_specified_name2850:$ 

_user_specified_name2848:$ 

_user_specified_name2846:$ 

_user_specified_name2844:$
 

_user_specified_name2842:$	 

_user_specified_name2840:$ 

_user_specified_name2838:$ 

_user_specified_name2836:$ 

_user_specified_name2834:$ 

_user_specified_name2832:$ 

_user_specified_name2830:$ 

_user_specified_name2828:$ 

_user_specified_name2826:XT
/
_output_shapes
:���������@@
!
_user_specified_name	input_4:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_3
�
�
G__inference_conv_rev3_4x4_layer_call_and_return_conditional_losses_5672

inputs8
conv2d_readvariableop_resource:A
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
H__inference_conv_prep0_3x3_layer_call_and_return_conditional_losses_4844

inputs8
conv2d_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
G__inference_conv_rev3_4x4_layer_call_and_return_conditional_losses_3475

inputs8
conv2d_readvariableop_resource:A
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
H__inference_conv_prep0_5x5_layer_call_and_return_conditional_losses_2317

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�w
�
A__inference_Decoder_layer_call_and_return_conditional_losses_3580
input_5,
conv_rev0_3x3_3289:2 
conv_rev0_3x3_3291:2,
conv_rev0_4x4_3305:
 
conv_rev0_4x4_3307:
,
conv_rev0_5x5_3321: 
conv_rev0_5x5_3323:,
conv_rev1_3x3_3346:A2 
conv_rev1_3x3_3348:2,
conv_rev1_4x4_3362:A
 
conv_rev1_4x4_3364:
,
conv_rev1_5x5_3378:A 
conv_rev1_5x5_3380:,
conv_rev2_3x3_3403:A2 
conv_rev2_3x3_3405:2,
conv_rev2_4x4_3419:A
 
conv_rev2_4x4_3421:
,
conv_rev2_5x5_3435:A 
conv_rev2_5x5_3437:,
conv_rev3_3x3_3460:A2 
conv_rev3_3x3_3462:2,
conv_rev3_4x4_3476:A
 
conv_rev3_4x4_3478:
,
conv_rev3_5x5_3492:A 
conv_rev3_5x5_3494:,
conv_rev4_3x3_3517:A2 
conv_rev4_3x3_3519:2,
conv_rev4_4x4_3533:A
 
conv_rev4_4x4_3535:
,
conv_rev5_5x5_3549:A 
conv_rev5_5x5_3551:'
output_s_3574:A
output_s_3576:
identity��%conv_rev0_3x3/StatefulPartitionedCall�%conv_rev0_4x4/StatefulPartitionedCall�%conv_rev0_5x5/StatefulPartitionedCall�%conv_rev1_3x3/StatefulPartitionedCall�%conv_rev1_4x4/StatefulPartitionedCall�%conv_rev1_5x5/StatefulPartitionedCall�%conv_rev2_3x3/StatefulPartitionedCall�%conv_rev2_4x4/StatefulPartitionedCall�%conv_rev2_5x5/StatefulPartitionedCall�%conv_rev3_3x3/StatefulPartitionedCall�%conv_rev3_4x4/StatefulPartitionedCall�%conv_rev3_5x5/StatefulPartitionedCall�%conv_rev4_3x3/StatefulPartitionedCall�%conv_rev4_4x4/StatefulPartitionedCall�%conv_rev5_5x5/StatefulPartitionedCall�&output_C_noise/StatefulPartitionedCall� output_S/StatefulPartitionedCall�
&output_C_noise/StatefulPartitionedCallStatefulPartitionedCallinput_5*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_output_C_noise_layer_call_and_return_conditional_losses_3276�
%conv_rev0_3x3/StatefulPartitionedCallStatefulPartitionedCall/output_C_noise/StatefulPartitionedCall:output:0conv_rev0_3x3_3289conv_rev0_3x3_3291*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev0_3x3_layer_call_and_return_conditional_losses_3288�
%conv_rev0_4x4/StatefulPartitionedCallStatefulPartitionedCall/output_C_noise/StatefulPartitionedCall:output:0conv_rev0_4x4_3305conv_rev0_4x4_3307*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev0_4x4_layer_call_and_return_conditional_losses_3304�
%conv_rev0_5x5/StatefulPartitionedCallStatefulPartitionedCall/output_C_noise/StatefulPartitionedCall:output:0conv_rev0_5x5_3321conv_rev0_5x5_3323*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev0_5x5_layer_call_and_return_conditional_losses_3320�
concatenate_8/PartitionedCallPartitionedCall.conv_rev0_3x3/StatefulPartitionedCall:output:0.conv_rev0_4x4/StatefulPartitionedCall:output:0.conv_rev0_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_8_layer_call_and_return_conditional_losses_3333�
%conv_rev1_3x3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0conv_rev1_3x3_3346conv_rev1_3x3_3348*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev1_3x3_layer_call_and_return_conditional_losses_3345�
%conv_rev1_4x4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0conv_rev1_4x4_3362conv_rev1_4x4_3364*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev1_4x4_layer_call_and_return_conditional_losses_3361�
%conv_rev1_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0conv_rev1_5x5_3378conv_rev1_5x5_3380*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev1_5x5_layer_call_and_return_conditional_losses_3377�
concatenate_9/PartitionedCallPartitionedCall.conv_rev1_3x3/StatefulPartitionedCall:output:0.conv_rev1_4x4/StatefulPartitionedCall:output:0.conv_rev1_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_9_layer_call_and_return_conditional_losses_3390�
%conv_rev2_3x3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0conv_rev2_3x3_3403conv_rev2_3x3_3405*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev2_3x3_layer_call_and_return_conditional_losses_3402�
%conv_rev2_4x4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0conv_rev2_4x4_3419conv_rev2_4x4_3421*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev2_4x4_layer_call_and_return_conditional_losses_3418�
%conv_rev2_5x5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0conv_rev2_5x5_3435conv_rev2_5x5_3437*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev2_5x5_layer_call_and_return_conditional_losses_3434�
concatenate_10/PartitionedCallPartitionedCall.conv_rev2_3x3/StatefulPartitionedCall:output:0.conv_rev2_4x4/StatefulPartitionedCall:output:0.conv_rev2_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_concatenate_10_layer_call_and_return_conditional_losses_3447�
%conv_rev3_3x3/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0conv_rev3_3x3_3460conv_rev3_3x3_3462*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev3_3x3_layer_call_and_return_conditional_losses_3459�
%conv_rev3_4x4/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0conv_rev3_4x4_3476conv_rev3_4x4_3478*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev3_4x4_layer_call_and_return_conditional_losses_3475�
%conv_rev3_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0conv_rev3_5x5_3492conv_rev3_5x5_3494*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev3_5x5_layer_call_and_return_conditional_losses_3491�
concatenate_11/PartitionedCallPartitionedCall.conv_rev3_3x3/StatefulPartitionedCall:output:0.conv_rev3_4x4/StatefulPartitionedCall:output:0.conv_rev3_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_concatenate_11_layer_call_and_return_conditional_losses_3504�
%conv_rev4_3x3/StatefulPartitionedCallStatefulPartitionedCall'concatenate_11/PartitionedCall:output:0conv_rev4_3x3_3517conv_rev4_3x3_3519*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev4_3x3_layer_call_and_return_conditional_losses_3516�
%conv_rev4_4x4/StatefulPartitionedCallStatefulPartitionedCall'concatenate_11/PartitionedCall:output:0conv_rev4_4x4_3533conv_rev4_4x4_3535*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev4_4x4_layer_call_and_return_conditional_losses_3532�
%conv_rev5_5x5/StatefulPartitionedCallStatefulPartitionedCall'concatenate_11/PartitionedCall:output:0conv_rev5_5x5_3549conv_rev5_5x5_3551*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_rev5_5x5_layer_call_and_return_conditional_losses_3548�
concatenate_12/PartitionedCallPartitionedCall.conv_rev4_3x3/StatefulPartitionedCall:output:0.conv_rev4_4x4/StatefulPartitionedCall:output:0.conv_rev5_5x5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@A* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_concatenate_12_layer_call_and_return_conditional_losses_3561�
 output_S/StatefulPartitionedCallStatefulPartitionedCall'concatenate_12/PartitionedCall:output:0output_s_3574output_s_3576*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_output_S_layer_call_and_return_conditional_losses_3573�
IdentityIdentity)output_S/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@�
NoOpNoOp&^conv_rev0_3x3/StatefulPartitionedCall&^conv_rev0_4x4/StatefulPartitionedCall&^conv_rev0_5x5/StatefulPartitionedCall&^conv_rev1_3x3/StatefulPartitionedCall&^conv_rev1_4x4/StatefulPartitionedCall&^conv_rev1_5x5/StatefulPartitionedCall&^conv_rev2_3x3/StatefulPartitionedCall&^conv_rev2_4x4/StatefulPartitionedCall&^conv_rev2_5x5/StatefulPartitionedCall&^conv_rev3_3x3/StatefulPartitionedCall&^conv_rev3_4x4/StatefulPartitionedCall&^conv_rev3_5x5/StatefulPartitionedCall&^conv_rev4_3x3/StatefulPartitionedCall&^conv_rev4_4x4/StatefulPartitionedCall&^conv_rev5_5x5/StatefulPartitionedCall'^output_C_noise/StatefulPartitionedCall!^output_S/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%conv_rev0_3x3/StatefulPartitionedCall%conv_rev0_3x3/StatefulPartitionedCall2N
%conv_rev0_4x4/StatefulPartitionedCall%conv_rev0_4x4/StatefulPartitionedCall2N
%conv_rev0_5x5/StatefulPartitionedCall%conv_rev0_5x5/StatefulPartitionedCall2N
%conv_rev1_3x3/StatefulPartitionedCall%conv_rev1_3x3/StatefulPartitionedCall2N
%conv_rev1_4x4/StatefulPartitionedCall%conv_rev1_4x4/StatefulPartitionedCall2N
%conv_rev1_5x5/StatefulPartitionedCall%conv_rev1_5x5/StatefulPartitionedCall2N
%conv_rev2_3x3/StatefulPartitionedCall%conv_rev2_3x3/StatefulPartitionedCall2N
%conv_rev2_4x4/StatefulPartitionedCall%conv_rev2_4x4/StatefulPartitionedCall2N
%conv_rev2_5x5/StatefulPartitionedCall%conv_rev2_5x5/StatefulPartitionedCall2N
%conv_rev3_3x3/StatefulPartitionedCall%conv_rev3_3x3/StatefulPartitionedCall2N
%conv_rev3_4x4/StatefulPartitionedCall%conv_rev3_4x4/StatefulPartitionedCall2N
%conv_rev3_5x5/StatefulPartitionedCall%conv_rev3_5x5/StatefulPartitionedCall2N
%conv_rev4_3x3/StatefulPartitionedCall%conv_rev4_3x3/StatefulPartitionedCall2N
%conv_rev4_4x4/StatefulPartitionedCall%conv_rev4_4x4/StatefulPartitionedCall2N
%conv_rev5_5x5/StatefulPartitionedCall%conv_rev5_5x5/StatefulPartitionedCall2P
&output_C_noise/StatefulPartitionedCall&output_C_noise/StatefulPartitionedCall2D
 output_S/StatefulPartitionedCall output_S/StatefulPartitionedCall:$  

_user_specified_name3576:$ 

_user_specified_name3574:$ 

_user_specified_name3551:$ 

_user_specified_name3549:$ 

_user_specified_name3535:$ 

_user_specified_name3533:$ 

_user_specified_name3519:$ 

_user_specified_name3517:$ 

_user_specified_name3494:$ 

_user_specified_name3492:$ 

_user_specified_name3478:$ 

_user_specified_name3476:$ 

_user_specified_name3462:$ 

_user_specified_name3460:$ 

_user_specified_name3437:$ 

_user_specified_name3435:$ 

_user_specified_name3421:$ 

_user_specified_name3419:$ 

_user_specified_name3405:$ 

_user_specified_name3403:$ 

_user_specified_name3380:$ 

_user_specified_name3378:$
 

_user_specified_name3364:$	 

_user_specified_name3362:$ 

_user_specified_name3348:$ 

_user_specified_name3346:$ 

_user_specified_name3323:$ 

_user_specified_name3321:$ 

_user_specified_name3307:$ 

_user_specified_name3305:$ 

_user_specified_name3291:$ 

_user_specified_name3289:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_5
�
�
-__inference_conv_prep1_4x4_layer_call_fn_4928

inputs!
unknown:A

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv_prep1_4x4_layer_call_and_return_conditional_losses_2358w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name4924:$ 

_user_specified_name4922:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_hid2_3x3_layer_call_and_return_conditional_losses_2521

inputs8
conv2d_readvariableop_resource:A2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_hid2_4x4_layer_call_and_return_conditional_losses_5177

inputs8
conv2d_readvariableop_resource:A
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_rev2_5x5_layer_call_and_return_conditional_losses_3434

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
G__inference_conv_rev1_4x4_layer_call_and_return_conditional_losses_3361

inputs8
conv2d_readvariableop_resource:A
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
f
-__inference_output_C_noise_layer_call_fn_5387

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_output_C_noise_layer_call_and_return_conditional_losses_3276w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
G__inference_conv_hid0_3x3_layer_call_and_return_conditional_losses_5007

inputs8
conv2d_readvariableop_resource:D2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:D2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@D: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@D
 
_user_specified_nameinputs
�
�
G__inference_conv_hid4_4x4_layer_call_and_return_conditional_losses_2651

inputs8
conv2d_readvariableop_resource:A
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A
*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@
S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
-__inference_conv_prep1_5x5_layer_call_fn_4948

inputs!
unknown:A
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv_prep1_5x5_layer_call_and_return_conditional_losses_2374w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name4944:$ 

_user_specified_name4942:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�%
�
&__inference_Encoder_layer_call_fn_3010
input_3
input_4!
unknown:2
	unknown_0:2#
	unknown_1:

	unknown_2:
#
	unknown_3:
	unknown_4:#
	unknown_5:A2
	unknown_6:2#
	unknown_7:A

	unknown_8:
#
	unknown_9:A

unknown_10:$

unknown_11:D2

unknown_12:2$

unknown_13:D


unknown_14:
$

unknown_15:D

unknown_16:$

unknown_17:A2

unknown_18:2$

unknown_19:A


unknown_20:
$

unknown_21:A

unknown_22:$

unknown_23:A2

unknown_24:2$

unknown_25:A


unknown_26:
$

unknown_27:A

unknown_28:$

unknown_29:A2

unknown_30:2$

unknown_31:A


unknown_32:
$

unknown_33:A

unknown_34:$

unknown_35:A2

unknown_36:2$

unknown_37:A


unknown_38:
$

unknown_39:A

unknown_40:$

unknown_41:A

unknown_42:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,-*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_Encoder_layer_call_and_return_conditional_losses_2822w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@@:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$- 

_user_specified_name3006:$, 

_user_specified_name3004:$+ 

_user_specified_name3002:$* 

_user_specified_name3000:$) 

_user_specified_name2998:$( 

_user_specified_name2996:$' 

_user_specified_name2994:$& 

_user_specified_name2992:$% 

_user_specified_name2990:$$ 

_user_specified_name2988:$# 

_user_specified_name2986:$" 

_user_specified_name2984:$! 

_user_specified_name2982:$  

_user_specified_name2980:$ 

_user_specified_name2978:$ 

_user_specified_name2976:$ 

_user_specified_name2974:$ 

_user_specified_name2972:$ 

_user_specified_name2970:$ 

_user_specified_name2968:$ 

_user_specified_name2966:$ 

_user_specified_name2964:$ 

_user_specified_name2962:$ 

_user_specified_name2960:$ 

_user_specified_name2958:$ 

_user_specified_name2956:$ 

_user_specified_name2954:$ 

_user_specified_name2952:$ 

_user_specified_name2950:$ 

_user_specified_name2948:$ 

_user_specified_name2946:$ 

_user_specified_name2944:$ 

_user_specified_name2942:$ 

_user_specified_name2940:$ 

_user_specified_name2938:$
 

_user_specified_name2936:$	 

_user_specified_name2934:$ 

_user_specified_name2932:$ 

_user_specified_name2930:$ 

_user_specified_name2928:$ 

_user_specified_name2926:$ 

_user_specified_name2924:$ 

_user_specified_name2922:$ 

_user_specified_name2920:XT
/
_output_shapes
:���������@@
!
_user_specified_name	input_4:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_3
�
�
H__inference_conv_prep1_5x5_layer_call_and_return_conditional_losses_4959

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
,__inference_conv_hid1_4x4_layer_call_fn_5091

inputs!
unknown:A

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_conv_hid1_4x4_layer_call_and_return_conditional_losses_2480w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5087:$ 

_user_specified_name5085:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�
�
'__inference_output_C_layer_call_fn_5371

inputs!
unknown:A
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_output_C_layer_call_and_return_conditional_losses_2692w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@A: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name5367:$ 

_user_specified_name5365:W S
/
_output_shapes
:���������@@A
 
_user_specified_nameinputs
�

G__inference_concatenate_1_layer_call_and_return_conditional_losses_2387

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@A_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������@@2:���������@@
:���������@@:WS
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������@@

 
_user_specified_nameinputs:W S
/
_output_shapes
:���������@@2
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������@@
C
input_28
serving_default_input_2:0���������@@J
concatenate_138
StatefulPartitionedCall:0���������@@tensorflow/serving/predict:ɒ	
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
�	
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
layer-8
layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
 layer_with_weights-11
 layer-17
!layer-18
"layer_with_weights-12
"layer-19
#layer_with_weights-13
#layer-20
$layer_with_weights-14
$layer-21
%layer-22
&layer_with_weights-15
&layer-23
'layer_with_weights-16
'layer-24
(layer_with_weights-17
(layer-25
)layer-26
*layer_with_weights-18
*layer-27
+layer_with_weights-19
+layer-28
,layer_with_weights-20
,layer-29
-layer-30
.layer_with_weights-21
.layer-31
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_network
�
5layer-0
6layer-1
7layer_with_weights-0
7layer-2
8layer_with_weights-1
8layer-3
9layer_with_weights-2
9layer-4
:layer-5
;layer_with_weights-3
;layer-6
<layer_with_weights-4
<layer-7
=layer_with_weights-5
=layer-8
>layer-9
?layer_with_weights-6
?layer-10
@layer_with_weights-7
@layer-11
Alayer_with_weights-8
Alayer-12
Blayer-13
Clayer_with_weights-9
Clayer-14
Dlayer_with_weights-10
Dlayer-15
Elayer_with_weights-11
Elayer-16
Flayer-17
Glayer_with_weights-12
Glayer-18
Hlayer_with_weights-13
Hlayer-19
Ilayer_with_weights-14
Ilayer-20
Jlayer-21
Klayer_with_weights-15
Klayer-22
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"
_tf_keras_network
�
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_layer
�
X0
Y1
Z2
[3
\4
]5
^6
_7
`8
a9
b10
c11
d12
e13
f14
g15
h16
i17
j18
k19
l20
m21
n22
o23
p24
q25
r26
s27
t28
u29
v30
w31
x32
y33
z34
{35
|36
}37
~38
39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72
�73
�74
�75"
trackable_list_wrapper
�
X0
Y1
Z2
[3
\4
]5
^6
_7
`8
a9
b10
c11
d12
e13
f14
g15
h16
i17
j18
k19
l20
m21
n22
o23
p24
q25
r26
s27
t28
u29
v30
w31
x32
y33
z34
{35
|36
}37
~38
39
�40
�41
�42
�43"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
$__inference_model_layer_call_fn_4486
$__inference_model_layer_call_fn_4644�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
?__inference_model_layer_call_and_return_conditional_losses_4168
?__inference_model_layer_call_and_return_conditional_losses_4328�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�B�
__inference__wrapped_model_2271input_1input_2"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
"
_tf_keras_input_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Xkernel
Ybias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Zkernel
[bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

\kernel
]bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

^kernel
_bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

`kernel
abias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

bkernel
cbias
!�_jit_compiled_convolution_op"
_tf_keras_layer
"
_tf_keras_input_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

dkernel
ebias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

fkernel
gbias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

hkernel
ibias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

jkernel
kbias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

lkernel
mbias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

nkernel
obias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

pkernel
qbias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

rkernel
sbias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

tkernel
ubias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

vkernel
wbias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

xkernel
ybias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

zkernel
{bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

|kernel
}bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

~kernel
bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
X0
Y1
Z2
[3
\4
]5
^6
_7
`8
a9
b10
c11
d12
e13
f14
g15
h16
i17
j18
k19
l20
m21
n22
o23
p24
q25
r26
s27
t28
u29
v30
w31
x32
y33
z34
{35
|36
}37
~38
39
�40
�41
�42
�43"
trackable_list_wrapper
�
X0
Y1
Z2
[3
\4
]5
^6
_7
`8
a9
b10
c11
d12
e13
f14
g15
h16
i17
j18
k19
l20
m21
n22
o23
p24
q25
r26
s27
t28
u29
v30
w31
x32
y33
z34
{35
|36
}37
~38
39
�40
�41
�42
�43"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
&__inference_Encoder_layer_call_fn_2916
&__inference_Encoder_layer_call_fn_3010�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
A__inference_Encoder_layer_call_and_return_conditional_losses_2699
A__inference_Encoder_layer_call_and_return_conditional_losses_2822�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_tf_keras_input_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
&__inference_Decoder_layer_call_fn_3743
&__inference_Decoder_layer_call_fn_3812�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
A__inference_Decoder_layer_call_and_return_conditional_losses_3580
A__inference_Decoder_layer_call_and_return_conditional_losses_3674�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_concatenate_13_layer_call_fn_4817�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_concatenate_13_layer_call_and_return_conditional_losses_4824�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
/:-22conv_prep0_3x3/kernel
!:22conv_prep0_3x3/bias
/:-
2conv_prep0_4x4/kernel
!:
2conv_prep0_4x4/bias
/:-2conv_prep0_5x5/kernel
!:2conv_prep0_5x5/bias
/:-A22conv_prep1_3x3/kernel
!:22conv_prep1_3x3/bias
/:-A
2conv_prep1_4x4/kernel
!:
2conv_prep1_4x4/bias
/:-A2conv_prep1_5x5/kernel
!:2conv_prep1_5x5/bias
.:,D22conv_hid0_3x3/kernel
 :22conv_hid0_3x3/bias
.:,D
2conv_hid0_4x4/kernel
 :
2conv_hid0_4x4/bias
.:,D2conv_hid0_5x5/kernel
 :2conv_hid0_5x5/bias
.:,A22conv_hid1_3x3/kernel
 :22conv_hid1_3x3/bias
.:,A
2conv_hid1_4x4/kernel
 :
2conv_hid1_4x4/bias
.:,A2conv_hid1_5x5/kernel
 :2conv_hid1_5x5/bias
.:,A22conv_hid2_3x3/kernel
 :22conv_hid2_3x3/bias
.:,A
2conv_hid2_4x4/kernel
 :
2conv_hid2_4x4/bias
.:,A2conv_hid2_5x5/kernel
 :2conv_hid2_5x5/bias
.:,A22conv_hid3_3x3/kernel
 :22conv_hid3_3x3/bias
.:,A
2conv_hid3_4x4/kernel
 :
2conv_hid3_4x4/bias
.:,A2conv_hid3_5x5/kernel
 :2conv_hid3_5x5/bias
.:,A22conv_hid4_3x3/kernel
 :22conv_hid4_3x3/bias
.:,A
2conv_hid4_4x4/kernel
 :
2conv_hid4_4x4/bias
.:,A2conv_hid5_5x5/kernel
 :2conv_hid5_5x5/bias
):'A2output_C/kernel
:2output_C/bias
.:,22conv_rev0_3x3/kernel
 :22conv_rev0_3x3/bias
.:,
2conv_rev0_4x4/kernel
 :
2conv_rev0_4x4/bias
.:,2conv_rev0_5x5/kernel
 :2conv_rev0_5x5/bias
.:,A22conv_rev1_3x3/kernel
 :22conv_rev1_3x3/bias
.:,A
2conv_rev1_4x4/kernel
 :
2conv_rev1_4x4/bias
.:,A2conv_rev1_5x5/kernel
 :2conv_rev1_5x5/bias
.:,A22conv_rev2_3x3/kernel
 :22conv_rev2_3x3/bias
.:,A
2conv_rev2_4x4/kernel
 :
2conv_rev2_4x4/bias
.:,A2conv_rev2_5x5/kernel
 :2conv_rev2_5x5/bias
.:,A22conv_rev3_3x3/kernel
 :22conv_rev3_3x3/bias
.:,A
2conv_rev3_4x4/kernel
 :
2conv_rev3_4x4/bias
.:,A2conv_rev3_5x5/kernel
 :2conv_rev3_5x5/bias
.:,A22conv_rev4_3x3/kernel
 :22conv_rev4_3x3/bias
.:,A
2conv_rev4_4x4/kernel
 :
2conv_rev4_4x4/bias
.:,A2conv_rev5_5x5/kernel
 :2conv_rev5_5x5/bias
):'A2output_S/kernel
:2output_S/bias
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_model_layer_call_fn_4486input_1input_2"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_model_layer_call_fn_4644input_1input_2"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_4168input_1input_2"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_4328input_1input_2"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72
�73
�74
�75
�76
�77
�78
�79
�80
�81
�82
�83
�84
�85
�86
�87
�88"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
"__inference_signature_wrapper_4811input_1input_2"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 '

kwonlyargs�
	jinput_1
	jinput_2
kwonlydefaults
 
annotations� *
 
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_conv_prep0_3x3_layer_call_fn_4833�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_conv_prep0_3x3_layer_call_and_return_conditional_losses_4844�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_conv_prep0_4x4_layer_call_fn_4853�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_conv_prep0_4x4_layer_call_and_return_conditional_losses_4864�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_conv_prep0_5x5_layer_call_fn_4873�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_conv_prep0_5x5_layer_call_and_return_conditional_losses_4884�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_concatenate_layer_call_fn_4891�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_concatenate_layer_call_and_return_conditional_losses_4899�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_conv_prep1_3x3_layer_call_fn_4908�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_conv_prep1_3x3_layer_call_and_return_conditional_losses_4919�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_conv_prep1_4x4_layer_call_fn_4928�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_conv_prep1_4x4_layer_call_and_return_conditional_losses_4939�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_conv_prep1_5x5_layer_call_fn_4948�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_conv_prep1_5x5_layer_call_and_return_conditional_losses_4959�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_concatenate_1_layer_call_fn_4966�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_concatenate_1_layer_call_and_return_conditional_losses_4974�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_concatenate_2_layer_call_fn_4980�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_concatenate_2_layer_call_and_return_conditional_losses_4987�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_hid0_3x3_layer_call_fn_4996�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_hid0_3x3_layer_call_and_return_conditional_losses_5007�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_hid0_4x4_layer_call_fn_5016�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_hid0_4x4_layer_call_and_return_conditional_losses_5027�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_hid0_5x5_layer_call_fn_5036�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_hid0_5x5_layer_call_and_return_conditional_losses_5047�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_concatenate_3_layer_call_fn_5054�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_concatenate_3_layer_call_and_return_conditional_losses_5062�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_hid1_3x3_layer_call_fn_5071�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_hid1_3x3_layer_call_and_return_conditional_losses_5082�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_hid1_4x4_layer_call_fn_5091�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_hid1_4x4_layer_call_and_return_conditional_losses_5102�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
n0
o1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_hid1_5x5_layer_call_fn_5111�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_hid1_5x5_layer_call_and_return_conditional_losses_5122�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_concatenate_4_layer_call_fn_5129�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_concatenate_4_layer_call_and_return_conditional_losses_5137�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
p0
q1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_hid2_3x3_layer_call_fn_5146�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_hid2_3x3_layer_call_and_return_conditional_losses_5157�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
r0
s1"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_hid2_4x4_layer_call_fn_5166�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_hid2_4x4_layer_call_and_return_conditional_losses_5177�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_hid2_5x5_layer_call_fn_5186�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_hid2_5x5_layer_call_and_return_conditional_losses_5197�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_concatenate_5_layer_call_fn_5204�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_concatenate_5_layer_call_and_return_conditional_losses_5212�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
v0
w1"
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_hid3_3x3_layer_call_fn_5221�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_hid3_3x3_layer_call_and_return_conditional_losses_5232�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_hid3_4x4_layer_call_fn_5241�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_hid3_4x4_layer_call_and_return_conditional_losses_5252�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_hid3_5x5_layer_call_fn_5261�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_hid3_5x5_layer_call_and_return_conditional_losses_5272�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_concatenate_6_layer_call_fn_5279�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_concatenate_6_layer_call_and_return_conditional_losses_5287�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
|0
}1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_hid4_3x3_layer_call_fn_5296�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_hid4_3x3_layer_call_and_return_conditional_losses_5307�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_hid4_4x4_layer_call_fn_5316�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_hid4_4x4_layer_call_and_return_conditional_losses_5327�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_hid5_5x5_layer_call_fn_5336�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_hid5_5x5_layer_call_and_return_conditional_losses_5347�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_concatenate_7_layer_call_fn_5354�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_concatenate_7_layer_call_and_return_conditional_losses_5362�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_output_C_layer_call_fn_5371�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_output_C_layer_call_and_return_conditional_losses_5382�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
 17
!18
"19
#20
$21
%22
&23
'24
(25
)26
*27
+28
,29
-30
.31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_Encoder_layer_call_fn_2916input_3input_4"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_Encoder_layer_call_fn_3010input_3input_4"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_Encoder_layer_call_and_return_conditional_losses_2699input_3input_4"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_Encoder_layer_call_and_return_conditional_losses_2822input_3input_4"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
-__inference_output_C_noise_layer_call_fn_5387
-__inference_output_C_noise_layer_call_fn_5392�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
H__inference_output_C_noise_layer_call_and_return_conditional_losses_5403
H__inference_output_C_noise_layer_call_and_return_conditional_losses_5407�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_rev0_3x3_layer_call_fn_5416�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_rev0_3x3_layer_call_and_return_conditional_losses_5427�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_rev0_4x4_layer_call_fn_5436�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_rev0_4x4_layer_call_and_return_conditional_losses_5447�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_rev0_5x5_layer_call_fn_5456�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_rev0_5x5_layer_call_and_return_conditional_losses_5467�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_concatenate_8_layer_call_fn_5474�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_concatenate_8_layer_call_and_return_conditional_losses_5482�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_rev1_3x3_layer_call_fn_5491�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_rev1_3x3_layer_call_and_return_conditional_losses_5502�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_rev1_4x4_layer_call_fn_5511�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_rev1_4x4_layer_call_and_return_conditional_losses_5522�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_rev1_5x5_layer_call_fn_5531�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_rev1_5x5_layer_call_and_return_conditional_losses_5542�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_concatenate_9_layer_call_fn_5549�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_concatenate_9_layer_call_and_return_conditional_losses_5557�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_rev2_3x3_layer_call_fn_5566�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_rev2_3x3_layer_call_and_return_conditional_losses_5577�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_rev2_4x4_layer_call_fn_5586�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_rev2_4x4_layer_call_and_return_conditional_losses_5597�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_rev2_5x5_layer_call_fn_5606�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_rev2_5x5_layer_call_and_return_conditional_losses_5617�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_concatenate_10_layer_call_fn_5624�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_concatenate_10_layer_call_and_return_conditional_losses_5632�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_rev3_3x3_layer_call_fn_5641�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_rev3_3x3_layer_call_and_return_conditional_losses_5652�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_rev3_4x4_layer_call_fn_5661�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_rev3_4x4_layer_call_and_return_conditional_losses_5672�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_rev3_5x5_layer_call_fn_5681�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_rev3_5x5_layer_call_and_return_conditional_losses_5692�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_concatenate_11_layer_call_fn_5699�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_concatenate_11_layer_call_and_return_conditional_losses_5707�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_rev4_3x3_layer_call_fn_5716�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_rev4_3x3_layer_call_and_return_conditional_losses_5727�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_rev4_4x4_layer_call_fn_5736�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_rev4_4x4_layer_call_and_return_conditional_losses_5747�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_rev5_5x5_layer_call_fn_5756�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_rev5_5x5_layer_call_and_return_conditional_losses_5767�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_concatenate_12_layer_call_fn_5774�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_concatenate_12_layer_call_and_return_conditional_losses_5782�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_output_S_layer_call_fn_5791�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_output_S_layer_call_and_return_conditional_losses_5802�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31"
trackable_list_wrapper
�
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
D15
E16
F17
G18
H19
I20
J21
K22"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_Decoder_layer_call_fn_3743input_5"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_Decoder_layer_call_fn_3812input_5"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_Decoder_layer_call_and_return_conditional_losses_3580input_5"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_Decoder_layer_call_and_return_conditional_losses_3674input_5"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_concatenate_13_layer_call_fn_4817inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_concatenate_13_layer_call_and_return_conditional_losses_4824inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
/:-22m/conv_prep0_3x3/kernel
/:-22v/conv_prep0_3x3/kernel
!:22m/conv_prep0_3x3/bias
!:22v/conv_prep0_3x3/bias
/:-
2m/conv_prep0_4x4/kernel
/:-
2v/conv_prep0_4x4/kernel
!:
2m/conv_prep0_4x4/bias
!:
2v/conv_prep0_4x4/bias
/:-2m/conv_prep0_5x5/kernel
/:-2v/conv_prep0_5x5/kernel
!:2m/conv_prep0_5x5/bias
!:2v/conv_prep0_5x5/bias
/:-A22m/conv_prep1_3x3/kernel
/:-A22v/conv_prep1_3x3/kernel
!:22m/conv_prep1_3x3/bias
!:22v/conv_prep1_3x3/bias
/:-A
2m/conv_prep1_4x4/kernel
/:-A
2v/conv_prep1_4x4/kernel
!:
2m/conv_prep1_4x4/bias
!:
2v/conv_prep1_4x4/bias
/:-A2m/conv_prep1_5x5/kernel
/:-A2v/conv_prep1_5x5/kernel
!:2m/conv_prep1_5x5/bias
!:2v/conv_prep1_5x5/bias
.:,D22m/conv_hid0_3x3/kernel
.:,D22v/conv_hid0_3x3/kernel
 :22m/conv_hid0_3x3/bias
 :22v/conv_hid0_3x3/bias
.:,D
2m/conv_hid0_4x4/kernel
.:,D
2v/conv_hid0_4x4/kernel
 :
2m/conv_hid0_4x4/bias
 :
2v/conv_hid0_4x4/bias
.:,D2m/conv_hid0_5x5/kernel
.:,D2v/conv_hid0_5x5/kernel
 :2m/conv_hid0_5x5/bias
 :2v/conv_hid0_5x5/bias
.:,A22m/conv_hid1_3x3/kernel
.:,A22v/conv_hid1_3x3/kernel
 :22m/conv_hid1_3x3/bias
 :22v/conv_hid1_3x3/bias
.:,A
2m/conv_hid1_4x4/kernel
.:,A
2v/conv_hid1_4x4/kernel
 :
2m/conv_hid1_4x4/bias
 :
2v/conv_hid1_4x4/bias
.:,A2m/conv_hid1_5x5/kernel
.:,A2v/conv_hid1_5x5/kernel
 :2m/conv_hid1_5x5/bias
 :2v/conv_hid1_5x5/bias
.:,A22m/conv_hid2_3x3/kernel
.:,A22v/conv_hid2_3x3/kernel
 :22m/conv_hid2_3x3/bias
 :22v/conv_hid2_3x3/bias
.:,A
2m/conv_hid2_4x4/kernel
.:,A
2v/conv_hid2_4x4/kernel
 :
2m/conv_hid2_4x4/bias
 :
2v/conv_hid2_4x4/bias
.:,A2m/conv_hid2_5x5/kernel
.:,A2v/conv_hid2_5x5/kernel
 :2m/conv_hid2_5x5/bias
 :2v/conv_hid2_5x5/bias
.:,A22m/conv_hid3_3x3/kernel
.:,A22v/conv_hid3_3x3/kernel
 :22m/conv_hid3_3x3/bias
 :22v/conv_hid3_3x3/bias
.:,A
2m/conv_hid3_4x4/kernel
.:,A
2v/conv_hid3_4x4/kernel
 :
2m/conv_hid3_4x4/bias
 :
2v/conv_hid3_4x4/bias
.:,A2m/conv_hid3_5x5/kernel
.:,A2v/conv_hid3_5x5/kernel
 :2m/conv_hid3_5x5/bias
 :2v/conv_hid3_5x5/bias
.:,A22m/conv_hid4_3x3/kernel
.:,A22v/conv_hid4_3x3/kernel
 :22m/conv_hid4_3x3/bias
 :22v/conv_hid4_3x3/bias
.:,A
2m/conv_hid4_4x4/kernel
.:,A
2v/conv_hid4_4x4/kernel
 :
2m/conv_hid4_4x4/bias
 :
2v/conv_hid4_4x4/bias
.:,A2m/conv_hid5_5x5/kernel
.:,A2v/conv_hid5_5x5/kernel
 :2m/conv_hid5_5x5/bias
 :2v/conv_hid5_5x5/bias
):'A2m/output_C/kernel
):'A2v/output_C/kernel
:2m/output_C/bias
:2v/output_C/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_conv_prep0_3x3_layer_call_fn_4833inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_conv_prep0_3x3_layer_call_and_return_conditional_losses_4844inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_conv_prep0_4x4_layer_call_fn_4853inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_conv_prep0_4x4_layer_call_and_return_conditional_losses_4864inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_conv_prep0_5x5_layer_call_fn_4873inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_conv_prep0_5x5_layer_call_and_return_conditional_losses_4884inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_concatenate_layer_call_fn_4891inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_concatenate_layer_call_and_return_conditional_losses_4899inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_conv_prep1_3x3_layer_call_fn_4908inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_conv_prep1_3x3_layer_call_and_return_conditional_losses_4919inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_conv_prep1_4x4_layer_call_fn_4928inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_conv_prep1_4x4_layer_call_and_return_conditional_losses_4939inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_conv_prep1_5x5_layer_call_fn_4948inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_conv_prep1_5x5_layer_call_and_return_conditional_losses_4959inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_concatenate_1_layer_call_fn_4966inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_concatenate_1_layer_call_and_return_conditional_losses_4974inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_concatenate_2_layer_call_fn_4980inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_concatenate_2_layer_call_and_return_conditional_losses_4987inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_hid0_3x3_layer_call_fn_4996inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_hid0_3x3_layer_call_and_return_conditional_losses_5007inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_hid0_4x4_layer_call_fn_5016inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_hid0_4x4_layer_call_and_return_conditional_losses_5027inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_hid0_5x5_layer_call_fn_5036inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_hid0_5x5_layer_call_and_return_conditional_losses_5047inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_concatenate_3_layer_call_fn_5054inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_concatenate_3_layer_call_and_return_conditional_losses_5062inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_hid1_3x3_layer_call_fn_5071inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_hid1_3x3_layer_call_and_return_conditional_losses_5082inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_hid1_4x4_layer_call_fn_5091inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_hid1_4x4_layer_call_and_return_conditional_losses_5102inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_hid1_5x5_layer_call_fn_5111inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_hid1_5x5_layer_call_and_return_conditional_losses_5122inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_concatenate_4_layer_call_fn_5129inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_concatenate_4_layer_call_and_return_conditional_losses_5137inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_hid2_3x3_layer_call_fn_5146inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_hid2_3x3_layer_call_and_return_conditional_losses_5157inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_hid2_4x4_layer_call_fn_5166inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_hid2_4x4_layer_call_and_return_conditional_losses_5177inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_hid2_5x5_layer_call_fn_5186inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_hid2_5x5_layer_call_and_return_conditional_losses_5197inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_concatenate_5_layer_call_fn_5204inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_concatenate_5_layer_call_and_return_conditional_losses_5212inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_hid3_3x3_layer_call_fn_5221inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_hid3_3x3_layer_call_and_return_conditional_losses_5232inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_hid3_4x4_layer_call_fn_5241inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_hid3_4x4_layer_call_and_return_conditional_losses_5252inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_hid3_5x5_layer_call_fn_5261inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_hid3_5x5_layer_call_and_return_conditional_losses_5272inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_concatenate_6_layer_call_fn_5279inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_concatenate_6_layer_call_and_return_conditional_losses_5287inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_hid4_3x3_layer_call_fn_5296inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_hid4_3x3_layer_call_and_return_conditional_losses_5307inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_hid4_4x4_layer_call_fn_5316inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_hid4_4x4_layer_call_and_return_conditional_losses_5327inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_hid5_5x5_layer_call_fn_5336inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_hid5_5x5_layer_call_and_return_conditional_losses_5347inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_concatenate_7_layer_call_fn_5354inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_concatenate_7_layer_call_and_return_conditional_losses_5362inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_output_C_layer_call_fn_5371inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_output_C_layer_call_and_return_conditional_losses_5382inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_output_C_noise_layer_call_fn_5387inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_output_C_noise_layer_call_fn_5392inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_output_C_noise_layer_call_and_return_conditional_losses_5403inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_output_C_noise_layer_call_and_return_conditional_losses_5407inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_rev0_3x3_layer_call_fn_5416inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_rev0_3x3_layer_call_and_return_conditional_losses_5427inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_rev0_4x4_layer_call_fn_5436inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_rev0_4x4_layer_call_and_return_conditional_losses_5447inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_rev0_5x5_layer_call_fn_5456inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_rev0_5x5_layer_call_and_return_conditional_losses_5467inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_concatenate_8_layer_call_fn_5474inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_concatenate_8_layer_call_and_return_conditional_losses_5482inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_rev1_3x3_layer_call_fn_5491inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_rev1_3x3_layer_call_and_return_conditional_losses_5502inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_rev1_4x4_layer_call_fn_5511inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_rev1_4x4_layer_call_and_return_conditional_losses_5522inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_rev1_5x5_layer_call_fn_5531inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_rev1_5x5_layer_call_and_return_conditional_losses_5542inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_concatenate_9_layer_call_fn_5549inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_concatenate_9_layer_call_and_return_conditional_losses_5557inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_rev2_3x3_layer_call_fn_5566inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_rev2_3x3_layer_call_and_return_conditional_losses_5577inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_rev2_4x4_layer_call_fn_5586inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_rev2_4x4_layer_call_and_return_conditional_losses_5597inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_rev2_5x5_layer_call_fn_5606inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_rev2_5x5_layer_call_and_return_conditional_losses_5617inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_concatenate_10_layer_call_fn_5624inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_concatenate_10_layer_call_and_return_conditional_losses_5632inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_rev3_3x3_layer_call_fn_5641inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_rev3_3x3_layer_call_and_return_conditional_losses_5652inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_rev3_4x4_layer_call_fn_5661inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_rev3_4x4_layer_call_and_return_conditional_losses_5672inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_rev3_5x5_layer_call_fn_5681inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_rev3_5x5_layer_call_and_return_conditional_losses_5692inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_concatenate_11_layer_call_fn_5699inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_concatenate_11_layer_call_and_return_conditional_losses_5707inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_rev4_3x3_layer_call_fn_5716inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_rev4_3x3_layer_call_and_return_conditional_losses_5727inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_rev4_4x4_layer_call_fn_5736inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_rev4_4x4_layer_call_and_return_conditional_losses_5747inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_rev5_5x5_layer_call_fn_5756inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_rev5_5x5_layer_call_and_return_conditional_losses_5767inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_concatenate_12_layer_call_fn_5774inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_concatenate_12_layer_call_and_return_conditional_losses_5782inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_output_S_layer_call_fn_5791inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_output_S_layer_call_and_return_conditional_losses_5802inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count�
A__inference_Decoder_layer_call_and_return_conditional_losses_3580�@��������������������������������@�=
6�3
)�&
input_5���������@@
p

 
� "4�1
*�'
tensor_0���������@@
� �
A__inference_Decoder_layer_call_and_return_conditional_losses_3674�@��������������������������������@�=
6�3
)�&
input_5���������@@
p 

 
� "4�1
*�'
tensor_0���������@@
� �
&__inference_Decoder_layer_call_fn_3743�@��������������������������������@�=
6�3
)�&
input_5���������@@
p

 
� ")�&
unknown���������@@�
&__inference_Decoder_layer_call_fn_3812�@��������������������������������@�=
6�3
)�&
input_5���������@@
p 

 
� ")�&
unknown���������@@�
A__inference_Encoder_layer_call_and_return_conditional_losses_2699�0XYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~����p�m
f�c
Y�V
)�&
input_3���������@@
)�&
input_4���������@@
p

 
� "4�1
*�'
tensor_0���������@@
� �
A__inference_Encoder_layer_call_and_return_conditional_losses_2822�0XYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~����p�m
f�c
Y�V
)�&
input_3���������@@
)�&
input_4���������@@
p 

 
� "4�1
*�'
tensor_0���������@@
� �
&__inference_Encoder_layer_call_fn_2916�0XYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~����p�m
f�c
Y�V
)�&
input_3���������@@
)�&
input_4���������@@
p

 
� ")�&
unknown���������@@�
&__inference_Encoder_layer_call_fn_3010�0XYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~����p�m
f�c
Y�V
)�&
input_3���������@@
)�&
input_4���������@@
p 

 
� ")�&
unknown���������@@�
__inference__wrapped_model_2271�pXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~������������������������������������h�e
^�[
Y�V
)�&
input_1���������@@
)�&
input_2���������@@
� "G�D
B
concatenate_130�-
concatenate_13���������@@�
H__inference_concatenate_10_layer_call_and_return_conditional_losses_5632����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� "4�1
*�'
tensor_0���������@@A
� �
-__inference_concatenate_10_layer_call_fn_5624����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� ")�&
unknown���������@@A�
H__inference_concatenate_11_layer_call_and_return_conditional_losses_5707����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� "4�1
*�'
tensor_0���������@@A
� �
-__inference_concatenate_11_layer_call_fn_5699����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� ")�&
unknown���������@@A�
H__inference_concatenate_12_layer_call_and_return_conditional_losses_5782����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� "4�1
*�'
tensor_0���������@@A
� �
-__inference_concatenate_12_layer_call_fn_5774����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� ")�&
unknown���������@@A�
H__inference_concatenate_13_layer_call_and_return_conditional_losses_4824�j�g
`�]
[�X
*�'
inputs_0���������@@
*�'
inputs_1���������@@
� "4�1
*�'
tensor_0���������@@
� �
-__inference_concatenate_13_layer_call_fn_4817�j�g
`�]
[�X
*�'
inputs_0���������@@
*�'
inputs_1���������@@
� ")�&
unknown���������@@�
G__inference_concatenate_1_layer_call_and_return_conditional_losses_4974����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� "4�1
*�'
tensor_0���������@@A
� �
,__inference_concatenate_1_layer_call_fn_4966����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� ")�&
unknown���������@@A�
G__inference_concatenate_2_layer_call_and_return_conditional_losses_4987�j�g
`�]
[�X
*�'
inputs_0���������@@
*�'
inputs_1���������@@A
� "4�1
*�'
tensor_0���������@@D
� �
,__inference_concatenate_2_layer_call_fn_4980�j�g
`�]
[�X
*�'
inputs_0���������@@
*�'
inputs_1���������@@A
� ")�&
unknown���������@@D�
G__inference_concatenate_3_layer_call_and_return_conditional_losses_5062����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� "4�1
*�'
tensor_0���������@@A
� �
,__inference_concatenate_3_layer_call_fn_5054����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� ")�&
unknown���������@@A�
G__inference_concatenate_4_layer_call_and_return_conditional_losses_5137����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� "4�1
*�'
tensor_0���������@@A
� �
,__inference_concatenate_4_layer_call_fn_5129����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� ")�&
unknown���������@@A�
G__inference_concatenate_5_layer_call_and_return_conditional_losses_5212����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� "4�1
*�'
tensor_0���������@@A
� �
,__inference_concatenate_5_layer_call_fn_5204����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� ")�&
unknown���������@@A�
G__inference_concatenate_6_layer_call_and_return_conditional_losses_5287����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� "4�1
*�'
tensor_0���������@@A
� �
,__inference_concatenate_6_layer_call_fn_5279����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� ")�&
unknown���������@@A�
G__inference_concatenate_7_layer_call_and_return_conditional_losses_5362����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� "4�1
*�'
tensor_0���������@@A
� �
,__inference_concatenate_7_layer_call_fn_5354����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� ")�&
unknown���������@@A�
G__inference_concatenate_8_layer_call_and_return_conditional_losses_5482����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� "4�1
*�'
tensor_0���������@@A
� �
,__inference_concatenate_8_layer_call_fn_5474����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� ")�&
unknown���������@@A�
G__inference_concatenate_9_layer_call_and_return_conditional_losses_5557����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� "4�1
*�'
tensor_0���������@@A
� �
,__inference_concatenate_9_layer_call_fn_5549����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� ")�&
unknown���������@@A�
E__inference_concatenate_layer_call_and_return_conditional_losses_4899����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� "4�1
*�'
tensor_0���������@@A
� �
*__inference_concatenate_layer_call_fn_4891����
���
���
*�'
inputs_0���������@@2
*�'
inputs_1���������@@

*�'
inputs_2���������@@
� ")�&
unknown���������@@A�
G__inference_conv_hid0_3x3_layer_call_and_return_conditional_losses_5007sde7�4
-�*
(�%
inputs���������@@D
� "4�1
*�'
tensor_0���������@@2
� �
,__inference_conv_hid0_3x3_layer_call_fn_4996hde7�4
-�*
(�%
inputs���������@@D
� ")�&
unknown���������@@2�
G__inference_conv_hid0_4x4_layer_call_and_return_conditional_losses_5027sfg7�4
-�*
(�%
inputs���������@@D
� "4�1
*�'
tensor_0���������@@

� �
,__inference_conv_hid0_4x4_layer_call_fn_5016hfg7�4
-�*
(�%
inputs���������@@D
� ")�&
unknown���������@@
�
G__inference_conv_hid0_5x5_layer_call_and_return_conditional_losses_5047shi7�4
-�*
(�%
inputs���������@@D
� "4�1
*�'
tensor_0���������@@
� �
,__inference_conv_hid0_5x5_layer_call_fn_5036hhi7�4
-�*
(�%
inputs���������@@D
� ")�&
unknown���������@@�
G__inference_conv_hid1_3x3_layer_call_and_return_conditional_losses_5082sjk7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@2
� �
,__inference_conv_hid1_3x3_layer_call_fn_5071hjk7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@2�
G__inference_conv_hid1_4x4_layer_call_and_return_conditional_losses_5102slm7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@

� �
,__inference_conv_hid1_4x4_layer_call_fn_5091hlm7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@
�
G__inference_conv_hid1_5x5_layer_call_and_return_conditional_losses_5122sno7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@
� �
,__inference_conv_hid1_5x5_layer_call_fn_5111hno7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@�
G__inference_conv_hid2_3x3_layer_call_and_return_conditional_losses_5157spq7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@2
� �
,__inference_conv_hid2_3x3_layer_call_fn_5146hpq7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@2�
G__inference_conv_hid2_4x4_layer_call_and_return_conditional_losses_5177srs7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@

� �
,__inference_conv_hid2_4x4_layer_call_fn_5166hrs7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@
�
G__inference_conv_hid2_5x5_layer_call_and_return_conditional_losses_5197stu7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@
� �
,__inference_conv_hid2_5x5_layer_call_fn_5186htu7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@�
G__inference_conv_hid3_3x3_layer_call_and_return_conditional_losses_5232svw7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@2
� �
,__inference_conv_hid3_3x3_layer_call_fn_5221hvw7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@2�
G__inference_conv_hid3_4x4_layer_call_and_return_conditional_losses_5252sxy7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@

� �
,__inference_conv_hid3_4x4_layer_call_fn_5241hxy7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@
�
G__inference_conv_hid3_5x5_layer_call_and_return_conditional_losses_5272sz{7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@
� �
,__inference_conv_hid3_5x5_layer_call_fn_5261hz{7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@�
G__inference_conv_hid4_3x3_layer_call_and_return_conditional_losses_5307s|}7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@2
� �
,__inference_conv_hid4_3x3_layer_call_fn_5296h|}7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@2�
G__inference_conv_hid4_4x4_layer_call_and_return_conditional_losses_5327s~7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@

� �
,__inference_conv_hid4_4x4_layer_call_fn_5316h~7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@
�
G__inference_conv_hid5_5x5_layer_call_and_return_conditional_losses_5347u��7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@
� �
,__inference_conv_hid5_5x5_layer_call_fn_5336j��7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@�
H__inference_conv_prep0_3x3_layer_call_and_return_conditional_losses_4844sXY7�4
-�*
(�%
inputs���������@@
� "4�1
*�'
tensor_0���������@@2
� �
-__inference_conv_prep0_3x3_layer_call_fn_4833hXY7�4
-�*
(�%
inputs���������@@
� ")�&
unknown���������@@2�
H__inference_conv_prep0_4x4_layer_call_and_return_conditional_losses_4864sZ[7�4
-�*
(�%
inputs���������@@
� "4�1
*�'
tensor_0���������@@

� �
-__inference_conv_prep0_4x4_layer_call_fn_4853hZ[7�4
-�*
(�%
inputs���������@@
� ")�&
unknown���������@@
�
H__inference_conv_prep0_5x5_layer_call_and_return_conditional_losses_4884s\]7�4
-�*
(�%
inputs���������@@
� "4�1
*�'
tensor_0���������@@
� �
-__inference_conv_prep0_5x5_layer_call_fn_4873h\]7�4
-�*
(�%
inputs���������@@
� ")�&
unknown���������@@�
H__inference_conv_prep1_3x3_layer_call_and_return_conditional_losses_4919s^_7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@2
� �
-__inference_conv_prep1_3x3_layer_call_fn_4908h^_7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@2�
H__inference_conv_prep1_4x4_layer_call_and_return_conditional_losses_4939s`a7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@

� �
-__inference_conv_prep1_4x4_layer_call_fn_4928h`a7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@
�
H__inference_conv_prep1_5x5_layer_call_and_return_conditional_losses_4959sbc7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@
� �
-__inference_conv_prep1_5x5_layer_call_fn_4948hbc7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@�
G__inference_conv_rev0_3x3_layer_call_and_return_conditional_losses_5427u��7�4
-�*
(�%
inputs���������@@
� "4�1
*�'
tensor_0���������@@2
� �
,__inference_conv_rev0_3x3_layer_call_fn_5416j��7�4
-�*
(�%
inputs���������@@
� ")�&
unknown���������@@2�
G__inference_conv_rev0_4x4_layer_call_and_return_conditional_losses_5447u��7�4
-�*
(�%
inputs���������@@
� "4�1
*�'
tensor_0���������@@

� �
,__inference_conv_rev0_4x4_layer_call_fn_5436j��7�4
-�*
(�%
inputs���������@@
� ")�&
unknown���������@@
�
G__inference_conv_rev0_5x5_layer_call_and_return_conditional_losses_5467u��7�4
-�*
(�%
inputs���������@@
� "4�1
*�'
tensor_0���������@@
� �
,__inference_conv_rev0_5x5_layer_call_fn_5456j��7�4
-�*
(�%
inputs���������@@
� ")�&
unknown���������@@�
G__inference_conv_rev1_3x3_layer_call_and_return_conditional_losses_5502u��7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@2
� �
,__inference_conv_rev1_3x3_layer_call_fn_5491j��7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@2�
G__inference_conv_rev1_4x4_layer_call_and_return_conditional_losses_5522u��7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@

� �
,__inference_conv_rev1_4x4_layer_call_fn_5511j��7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@
�
G__inference_conv_rev1_5x5_layer_call_and_return_conditional_losses_5542u��7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@
� �
,__inference_conv_rev1_5x5_layer_call_fn_5531j��7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@�
G__inference_conv_rev2_3x3_layer_call_and_return_conditional_losses_5577u��7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@2
� �
,__inference_conv_rev2_3x3_layer_call_fn_5566j��7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@2�
G__inference_conv_rev2_4x4_layer_call_and_return_conditional_losses_5597u��7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@

� �
,__inference_conv_rev2_4x4_layer_call_fn_5586j��7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@
�
G__inference_conv_rev2_5x5_layer_call_and_return_conditional_losses_5617u��7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@
� �
,__inference_conv_rev2_5x5_layer_call_fn_5606j��7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@�
G__inference_conv_rev3_3x3_layer_call_and_return_conditional_losses_5652u��7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@2
� �
,__inference_conv_rev3_3x3_layer_call_fn_5641j��7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@2�
G__inference_conv_rev3_4x4_layer_call_and_return_conditional_losses_5672u��7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@

� �
,__inference_conv_rev3_4x4_layer_call_fn_5661j��7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@
�
G__inference_conv_rev3_5x5_layer_call_and_return_conditional_losses_5692u��7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@
� �
,__inference_conv_rev3_5x5_layer_call_fn_5681j��7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@�
G__inference_conv_rev4_3x3_layer_call_and_return_conditional_losses_5727u��7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@2
� �
,__inference_conv_rev4_3x3_layer_call_fn_5716j��7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@2�
G__inference_conv_rev4_4x4_layer_call_and_return_conditional_losses_5747u��7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@

� �
,__inference_conv_rev4_4x4_layer_call_fn_5736j��7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@
�
G__inference_conv_rev5_5x5_layer_call_and_return_conditional_losses_5767u��7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@
� �
,__inference_conv_rev5_5x5_layer_call_fn_5756j��7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@�
?__inference_model_layer_call_and_return_conditional_losses_4168�pXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~������������������������������������p�m
f�c
Y�V
)�&
input_1���������@@
)�&
input_2���������@@
p

 
� "4�1
*�'
tensor_0���������@@
� �
?__inference_model_layer_call_and_return_conditional_losses_4328�pXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~������������������������������������p�m
f�c
Y�V
)�&
input_1���������@@
)�&
input_2���������@@
p 

 
� "4�1
*�'
tensor_0���������@@
� �
$__inference_model_layer_call_fn_4486�pXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~������������������������������������p�m
f�c
Y�V
)�&
input_1���������@@
)�&
input_2���������@@
p

 
� ")�&
unknown���������@@�
$__inference_model_layer_call_fn_4644�pXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~������������������������������������p�m
f�c
Y�V
)�&
input_1���������@@
)�&
input_2���������@@
p 

 
� ")�&
unknown���������@@�
B__inference_output_C_layer_call_and_return_conditional_losses_5382u��7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@
� �
'__inference_output_C_layer_call_fn_5371j��7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@�
H__inference_output_C_noise_layer_call_and_return_conditional_losses_5403s;�8
1�.
(�%
inputs���������@@
p
� "4�1
*�'
tensor_0���������@@
� �
H__inference_output_C_noise_layer_call_and_return_conditional_losses_5407s;�8
1�.
(�%
inputs���������@@
p 
� "4�1
*�'
tensor_0���������@@
� �
-__inference_output_C_noise_layer_call_fn_5387h;�8
1�.
(�%
inputs���������@@
p
� ")�&
unknown���������@@�
-__inference_output_C_noise_layer_call_fn_5392h;�8
1�.
(�%
inputs���������@@
p 
� ")�&
unknown���������@@�
B__inference_output_S_layer_call_and_return_conditional_losses_5802u��7�4
-�*
(�%
inputs���������@@A
� "4�1
*�'
tensor_0���������@@
� �
'__inference_output_S_layer_call_fn_5791j��7�4
-�*
(�%
inputs���������@@A
� ")�&
unknown���������@@�
"__inference_signature_wrapper_4811�pXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~������������������������������������y�v
� 
o�l
4
input_1)�&
input_1���������@@
4
input_2)�&
input_2���������@@"G�D
B
concatenate_130�-
concatenate_13���������@@