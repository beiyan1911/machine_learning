­)
°


:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
H
ShardedFilename
basename	
shard

num_shards
filename
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "
saved_test*1.8.02
b'unknown'8
P
input-xPlaceholder*
shape:*
dtype0*
_output_shapes
:

"w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB:*
_class

loc:@w
{
 w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class

loc:@w
{
 w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  ?*
_class

loc:@w
§
*w/Initializer/random_uniform/RandomUniformRandomUniform"w/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:*
T0*
_class

loc:@w
¢
 w/Initializer/random_uniform/subSub w/Initializer/random_uniform/max w/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class

loc:@w
°
 w/Initializer/random_uniform/mulMul*w/Initializer/random_uniform/RandomUniform w/Initializer/random_uniform/sub*
_output_shapes
:*
T0*
_class

loc:@w
¢
w/Initializer/random_uniformAdd w/Initializer/random_uniform/mul w/Initializer/random_uniform/min*
_output_shapes
:*
T0*
_class

loc:@w
_
w
VariableV2*
_class

loc:@w*
shape:*
dtype0*
_output_shapes
:
n
w/AssignAssignww/Initializer/random_uniform*
_output_shapes
:*
T0*
_class

loc:@w
P
w/readIdentityw*
T0*
_class

loc:@w*
_output_shapes
:
E
output-zAddinput-xw/read*
T0*
_output_shapes
:

initNoOp	^w/Assign
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_b2665ed88ca64cf6813006d732725a31/part*
dtype0*
_output_shapes
: 
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
q
save/SaveV2/tensor_namesConst"/device:CPU:0*
valueBBw*
dtype0*
_output_shapes
:
t
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesw"/device:CPU:0*
dtypes
2
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
 
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*
N*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
t
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueBBw
w
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
c
save/AssignAssignwsave/RestoreV2*
T0*
_class

loc:@w*
_output_shapes
:
(
save/restore_shardNoOp^save/Assign
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"J
	variables=;
9
w:0w/Assignw/read:02w/Initializer/random_uniform:0"T
trainable_variables=;
9
w:0w/Assignw/read:02w/Initializer/random_uniform:0