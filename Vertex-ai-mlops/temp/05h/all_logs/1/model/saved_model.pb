٨
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:?*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:?*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:?*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:?*
dtype0
?
&SGD/batch_normalization/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&SGD/batch_normalization/gamma/momentum
?
:SGD/batch_normalization/gamma/momentum/Read/ReadVariableOpReadVariableOp&SGD/batch_normalization/gamma/momentum*
_output_shapes
:*
dtype0
?
%SGD/batch_normalization/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%SGD/batch_normalization/beta/momentum
?
9SGD/batch_normalization/beta/momentum/Read/ReadVariableOpReadVariableOp%SGD/batch_normalization/beta/momentum*
_output_shapes
:*
dtype0
?
SGD/dense/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameSGD/dense/kernel/momentum
?
-SGD/dense/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense/kernel/momentum*
_output_shapes

:*
dtype0
?
SGD/dense/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameSGD/dense/bias/momentum

+SGD/dense/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense/bias/momentum*
_output_shapes
:*
dtype0

NoOpNoOp
?'
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?'
value?'B?' B?'
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer_with_weights-0
 layer-31
!layer_with_weights-1
!layer-32
"	optimizer
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'
signatures
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
x
(_feature_columns
)
_resources
*	variables
+trainable_variables
,regularization_losses
-	keras_api
?
.axis
	/gamma
0beta
1moving_mean
2moving_variance
3	variables
4trainable_variables
5regularization_losses
6	keras_api
h

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
v
=iter
	>decay
?learning_rate
@momentum/momentumg0momentumh7momentumi8momentumj
*
/0
01
12
23
74
85

/0
01
72
83
 
?
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
#	variables
$trainable_variables
%regularization_losses
 
 
 
 
 
 
?
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
*	variables
+trainable_variables
,regularization_losses
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

/0
01
12
23

/0
01
 
?
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
3	variables
4trainable_variables
5regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

70
81

70
81
 
?
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
9	variables
:trainable_variables
;regularization_losses
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

10
21
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32

U0
V1
W2
 
 
 
 
 
 
 

10
21
 
 
 
 
 
 
 
 
 
4
	Xtotal
	Ycount
Z	variables
[	keras_api
D
	\total
	]count
^
_fn_kwargs
_	variables
`	keras_api
p
atrue_positives
btrue_negatives
cfalse_positives
dfalse_negatives
e	variables
f	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

X0
Y1

Z	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

\0
]1

_	variables
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

a0
b1
c2
d3

e	variables
??
VARIABLE_VALUE&SGD/batch_normalization/gamma/momentumXlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE%SGD/batch_normalization/beta/momentumWlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
y
serving_default_AmountPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
w
serving_default_TimePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
u
serving_default_V1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_V10Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_V11Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_V12Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_V13Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_V14Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_V15Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_V16Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_V17Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_V18Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_V19Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
u
serving_default_V2Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_V20Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_V21Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_V22Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_V23Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_V24Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_V25Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_V26Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_V27Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_V28Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
u
serving_default_V3Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
u
serving_default_V4Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
u
serving_default_V5Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
u
serving_default_V6Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
u
serving_default_V7Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
u
serving_default_V8Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
u
serving_default_V9Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_Amountserving_default_Timeserving_default_V1serving_default_V10serving_default_V11serving_default_V12serving_default_V13serving_default_V14serving_default_V15serving_default_V16serving_default_V17serving_default_V18serving_default_V19serving_default_V2serving_default_V20serving_default_V21serving_default_V22serving_default_V23serving_default_V24serving_default_V25serving_default_V26serving_default_V27serving_default_V28serving_default_V3serving_default_V4serving_default_V5serving_default_V6serving_default_V7serving_default_V8serving_default_V9#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense/kernel
dense/bias*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

 !"#*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_140622
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp:SGD/batch_normalization/gamma/momentum/Read/ReadVariableOp9SGD/batch_normalization/beta/momentum/Read/ReadVariableOp-SGD/dense/kernel/momentum/Read/ReadVariableOp+SGD/dense/bias/momentum/Read/ReadVariableOpConst*#
Tin
2	*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_142160
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense/kernel
dense/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttotal_1count_1true_positivestrue_negativesfalse_positivesfalse_negatives&SGD/batch_normalization/gamma/momentum%SGD/batch_normalization/beta/momentumSGD/dense/kernel/momentumSGD/dense/bias/momentum*"
Tin
2*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_142236??
?%
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_139539

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?&
?
&__inference_model_layer_call_fn_140714
inputs_amount
inputs_time
	inputs_v1

inputs_v10

inputs_v11

inputs_v12

inputs_v13

inputs_v14

inputs_v15

inputs_v16

inputs_v17

inputs_v18

inputs_v19
	inputs_v2

inputs_v20

inputs_v21

inputs_v22

inputs_v23

inputs_v24

inputs_v25

inputs_v26

inputs_v27

inputs_v28
	inputs_v3
	inputs_v4
	inputs_v5
	inputs_v6
	inputs_v7
	inputs_v8
	inputs_v9
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_amountinputs_time	inputs_v1
inputs_v10
inputs_v11
inputs_v12
inputs_v13
inputs_v14
inputs_v15
inputs_v16
inputs_v17
inputs_v18
inputs_v19	inputs_v2
inputs_v20
inputs_v21
inputs_v22
inputs_v23
inputs_v24
inputs_v25
inputs_v26
inputs_v27
inputs_v28	inputs_v3	inputs_v4	inputs_v5	inputs_v6	inputs_v7	inputs_v8	inputs_v9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
 !"#*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_140413o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_nameinputs/Amount:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/Time:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V1:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V10:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V11:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V12:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V13:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V14:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V15:S	O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V16:S
O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V17:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V18:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V19:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V2:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V20:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V21:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V22:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V23:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V24:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V25:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V26:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V27:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V28:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V3:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V4:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V5:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V6:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V7:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V8:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V9
??
?
J__inference_dense_features_layer_call_and_return_conditional_losses_139890
features

features_1

features_2

features_3

features_4

features_5

features_6

features_7

features_8

features_9
features_10
features_11
features_12
features_13
features_14
features_15
features_16
features_17
features_18
features_19
features_20
features_21
features_22
features_23
features_24
features_25
features_26
features_27
features_28
features_29
identityD
Amount/ShapeShapefeatures*
T0*
_output_shapes
:d
Amount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
Amount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
Amount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Amount/strided_sliceStridedSliceAmount/Shape:output:0#Amount/strided_slice/stack:output:0%Amount/strided_slice/stack_1:output:0%Amount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Amount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Amount/Reshape/shapePackAmount/strided_slice:output:0Amount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:t
Amount/ReshapeReshapefeaturesAmount/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D

Time/ShapeShape
features_1*
T0*
_output_shapes
:b
Time/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
Time/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
Time/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Time/strided_sliceStridedSliceTime/Shape:output:0!Time/strided_slice/stack:output:0#Time/strided_slice/stack_1:output:0#Time/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
Time/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Time/Reshape/shapePackTime/strided_slice:output:0Time/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
Time/ReshapeReshape
features_1Time/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????B
V1/ShapeShape
features_2*
T0*
_output_shapes
:`
V1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V1/strided_sliceStridedSliceV1/Shape:output:0V1/strided_slice/stack:output:0!V1/strided_slice/stack_1:output:0!V1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V1/Reshape/shapePackV1/strided_slice:output:0V1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:n

V1/ReshapeReshape
features_2V1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
	V10/ShapeShape
features_3*
T0*
_output_shapes
:a
V10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V10/strided_sliceStridedSliceV10/Shape:output:0 V10/strided_slice/stack:output:0"V10/strided_slice/stack_1:output:0"V10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V10/Reshape/shapePackV10/strided_slice:output:0V10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:p
V10/ReshapeReshape
features_3V10/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
	V11/ShapeShape
features_4*
T0*
_output_shapes
:a
V11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V11/strided_sliceStridedSliceV11/Shape:output:0 V11/strided_slice/stack:output:0"V11/strided_slice/stack_1:output:0"V11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V11/Reshape/shapePackV11/strided_slice:output:0V11/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:p
V11/ReshapeReshape
features_4V11/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
	V12/ShapeShape
features_5*
T0*
_output_shapes
:a
V12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V12/strided_sliceStridedSliceV12/Shape:output:0 V12/strided_slice/stack:output:0"V12/strided_slice/stack_1:output:0"V12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V12/Reshape/shapePackV12/strided_slice:output:0V12/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:p
V12/ReshapeReshape
features_5V12/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
	V13/ShapeShape
features_6*
T0*
_output_shapes
:a
V13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V13/strided_sliceStridedSliceV13/Shape:output:0 V13/strided_slice/stack:output:0"V13/strided_slice/stack_1:output:0"V13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V13/Reshape/shapePackV13/strided_slice:output:0V13/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:p
V13/ReshapeReshape
features_6V13/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
	V14/ShapeShape
features_7*
T0*
_output_shapes
:a
V14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V14/strided_sliceStridedSliceV14/Shape:output:0 V14/strided_slice/stack:output:0"V14/strided_slice/stack_1:output:0"V14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V14/Reshape/shapePackV14/strided_slice:output:0V14/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:p
V14/ReshapeReshape
features_7V14/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
	V15/ShapeShape
features_8*
T0*
_output_shapes
:a
V15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V15/strided_sliceStridedSliceV15/Shape:output:0 V15/strided_slice/stack:output:0"V15/strided_slice/stack_1:output:0"V15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V15/Reshape/shapePackV15/strided_slice:output:0V15/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:p
V15/ReshapeReshape
features_8V15/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
	V16/ShapeShape
features_9*
T0*
_output_shapes
:a
V16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V16/strided_sliceStridedSliceV16/Shape:output:0 V16/strided_slice/stack:output:0"V16/strided_slice/stack_1:output:0"V16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V16/Reshape/shapePackV16/strided_slice:output:0V16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:p
V16/ReshapeReshape
features_9V16/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V17/ShapeShapefeatures_10*
T0*
_output_shapes
:a
V17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V17/strided_sliceStridedSliceV17/Shape:output:0 V17/strided_slice/stack:output:0"V17/strided_slice/stack_1:output:0"V17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V17/Reshape/shapePackV17/strided_slice:output:0V17/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V17/ReshapeReshapefeatures_10V17/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V18/ShapeShapefeatures_11*
T0*
_output_shapes
:a
V18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V18/strided_sliceStridedSliceV18/Shape:output:0 V18/strided_slice/stack:output:0"V18/strided_slice/stack_1:output:0"V18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V18/Reshape/shapePackV18/strided_slice:output:0V18/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V18/ReshapeReshapefeatures_11V18/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V19/ShapeShapefeatures_12*
T0*
_output_shapes
:a
V19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V19/strided_sliceStridedSliceV19/Shape:output:0 V19/strided_slice/stack:output:0"V19/strided_slice/stack_1:output:0"V19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V19/Reshape/shapePackV19/strided_slice:output:0V19/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V19/ReshapeReshapefeatures_12V19/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V2/ShapeShapefeatures_13*
T0*
_output_shapes
:`
V2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V2/strided_sliceStridedSliceV2/Shape:output:0V2/strided_slice/stack:output:0!V2/strided_slice/stack_1:output:0!V2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V2/Reshape/shapePackV2/strided_slice:output:0V2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V2/ReshapeReshapefeatures_13V2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V20/ShapeShapefeatures_14*
T0*
_output_shapes
:a
V20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V20/strided_sliceStridedSliceV20/Shape:output:0 V20/strided_slice/stack:output:0"V20/strided_slice/stack_1:output:0"V20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V20/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V20/Reshape/shapePackV20/strided_slice:output:0V20/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V20/ReshapeReshapefeatures_14V20/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V21/ShapeShapefeatures_15*
T0*
_output_shapes
:a
V21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V21/strided_sliceStridedSliceV21/Shape:output:0 V21/strided_slice/stack:output:0"V21/strided_slice/stack_1:output:0"V21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V21/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V21/Reshape/shapePackV21/strided_slice:output:0V21/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V21/ReshapeReshapefeatures_15V21/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V22/ShapeShapefeatures_16*
T0*
_output_shapes
:a
V22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V22/strided_sliceStridedSliceV22/Shape:output:0 V22/strided_slice/stack:output:0"V22/strided_slice/stack_1:output:0"V22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V22/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V22/Reshape/shapePackV22/strided_slice:output:0V22/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V22/ReshapeReshapefeatures_16V22/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V23/ShapeShapefeatures_17*
T0*
_output_shapes
:a
V23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V23/strided_sliceStridedSliceV23/Shape:output:0 V23/strided_slice/stack:output:0"V23/strided_slice/stack_1:output:0"V23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V23/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V23/Reshape/shapePackV23/strided_slice:output:0V23/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V23/ReshapeReshapefeatures_17V23/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V24/ShapeShapefeatures_18*
T0*
_output_shapes
:a
V24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V24/strided_sliceStridedSliceV24/Shape:output:0 V24/strided_slice/stack:output:0"V24/strided_slice/stack_1:output:0"V24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V24/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V24/Reshape/shapePackV24/strided_slice:output:0V24/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V24/ReshapeReshapefeatures_18V24/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V25/ShapeShapefeatures_19*
T0*
_output_shapes
:a
V25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V25/strided_sliceStridedSliceV25/Shape:output:0 V25/strided_slice/stack:output:0"V25/strided_slice/stack_1:output:0"V25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V25/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V25/Reshape/shapePackV25/strided_slice:output:0V25/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V25/ReshapeReshapefeatures_19V25/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V26/ShapeShapefeatures_20*
T0*
_output_shapes
:a
V26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V26/strided_sliceStridedSliceV26/Shape:output:0 V26/strided_slice/stack:output:0"V26/strided_slice/stack_1:output:0"V26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V26/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V26/Reshape/shapePackV26/strided_slice:output:0V26/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V26/ReshapeReshapefeatures_20V26/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V27/ShapeShapefeatures_21*
T0*
_output_shapes
:a
V27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V27/strided_sliceStridedSliceV27/Shape:output:0 V27/strided_slice/stack:output:0"V27/strided_slice/stack_1:output:0"V27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V27/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V27/Reshape/shapePackV27/strided_slice:output:0V27/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V27/ReshapeReshapefeatures_21V27/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V28/ShapeShapefeatures_22*
T0*
_output_shapes
:a
V28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V28/strided_sliceStridedSliceV28/Shape:output:0 V28/strided_slice/stack:output:0"V28/strided_slice/stack_1:output:0"V28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V28/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V28/Reshape/shapePackV28/strided_slice:output:0V28/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V28/ReshapeReshapefeatures_22V28/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V3/ShapeShapefeatures_23*
T0*
_output_shapes
:`
V3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V3/strided_sliceStridedSliceV3/Shape:output:0V3/strided_slice/stack:output:0!V3/strided_slice/stack_1:output:0!V3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V3/Reshape/shapePackV3/strided_slice:output:0V3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V3/ReshapeReshapefeatures_23V3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V4/ShapeShapefeatures_24*
T0*
_output_shapes
:`
V4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V4/strided_sliceStridedSliceV4/Shape:output:0V4/strided_slice/stack:output:0!V4/strided_slice/stack_1:output:0!V4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V4/Reshape/shapePackV4/strided_slice:output:0V4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V4/ReshapeReshapefeatures_24V4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V5/ShapeShapefeatures_25*
T0*
_output_shapes
:`
V5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V5/strided_sliceStridedSliceV5/Shape:output:0V5/strided_slice/stack:output:0!V5/strided_slice/stack_1:output:0!V5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V5/Reshape/shapePackV5/strided_slice:output:0V5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V5/ReshapeReshapefeatures_25V5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V6/ShapeShapefeatures_26*
T0*
_output_shapes
:`
V6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V6/strided_sliceStridedSliceV6/Shape:output:0V6/strided_slice/stack:output:0!V6/strided_slice/stack_1:output:0!V6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V6/Reshape/shapePackV6/strided_slice:output:0V6/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V6/ReshapeReshapefeatures_26V6/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V7/ShapeShapefeatures_27*
T0*
_output_shapes
:`
V7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V7/strided_sliceStridedSliceV7/Shape:output:0V7/strided_slice/stack:output:0!V7/strided_slice/stack_1:output:0!V7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V7/Reshape/shapePackV7/strided_slice:output:0V7/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V7/ReshapeReshapefeatures_27V7/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V8/ShapeShapefeatures_28*
T0*
_output_shapes
:`
V8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V8/strided_sliceStridedSliceV8/Shape:output:0V8/strided_slice/stack:output:0!V8/strided_slice/stack_1:output:0!V8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V8/Reshape/shapePackV8/strided_slice:output:0V8/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V8/ReshapeReshapefeatures_28V8/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V9/ShapeShapefeatures_29*
T0*
_output_shapes
:`
V9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V9/strided_sliceStridedSliceV9/Shape:output:0V9/strided_slice/stack:output:0!V9/strided_slice/stack_1:output:0!V9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V9/Reshape/shapePackV9/strided_slice:output:0V9/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V9/ReshapeReshapefeatures_29V9/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2Amount/Reshape:output:0Time/Reshape:output:0V1/Reshape:output:0V10/Reshape:output:0V11/Reshape:output:0V12/Reshape:output:0V13/Reshape:output:0V14/Reshape:output:0V15/Reshape:output:0V16/Reshape:output:0V17/Reshape:output:0V18/Reshape:output:0V19/Reshape:output:0V2/Reshape:output:0V20/Reshape:output:0V21/Reshape:output:0V22/Reshape:output:0V23/Reshape:output:0V24/Reshape:output:0V25/Reshape:output:0V26/Reshape:output:0V27/Reshape:output:0V28/Reshape:output:0V3/Reshape:output:0V4/Reshape:output:0V5/Reshape:output:0V6/Reshape:output:0V7/Reshape:output:0V8/Reshape:output:0V9/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
features:Q
M
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features
??
?

A__inference_model_layer_call_and_return_conditional_losses_141324
inputs_amount
inputs_time
	inputs_v1

inputs_v10

inputs_v11

inputs_v12

inputs_v13

inputs_v14

inputs_v15

inputs_v16

inputs_v17

inputs_v18

inputs_v19
	inputs_v2

inputs_v20

inputs_v21

inputs_v22

inputs_v23

inputs_v24

inputs_v25

inputs_v26

inputs_v27

inputs_v28
	inputs_v3
	inputs_v4
	inputs_v5
	inputs_v6
	inputs_v7
	inputs_v8
	inputs_v9I
;batch_normalization_assignmovingavg_readvariableop_resource:K
=batch_normalization_assignmovingavg_1_readvariableop_resource:G
9batch_normalization_batchnorm_mul_readvariableop_resource:C
5batch_normalization_batchnorm_readvariableop_resource:6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity??#batch_normalization/AssignMovingAvg?2batch_normalization/AssignMovingAvg/ReadVariableOp?%batch_normalization/AssignMovingAvg_1?4batch_normalization/AssignMovingAvg_1/ReadVariableOp?,batch_normalization/batchnorm/ReadVariableOp?0batch_normalization/batchnorm/mul/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOpX
dense_features/Amount/ShapeShapeinputs_amount*
T0*
_output_shapes
:s
)dense_features/Amount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+dense_features/Amount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+dense_features/Amount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#dense_features/Amount/strided_sliceStridedSlice$dense_features/Amount/Shape:output:02dense_features/Amount/strided_slice/stack:output:04dense_features/Amount/strided_slice/stack_1:output:04dense_features/Amount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%dense_features/Amount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#dense_features/Amount/Reshape/shapePack,dense_features/Amount/strided_slice:output:0.dense_features/Amount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/Amount/ReshapeReshapeinputs_amount,dense_features/Amount/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????T
dense_features/Time/ShapeShapeinputs_time*
T0*
_output_shapes
:q
'dense_features/Time/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)dense_features/Time/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)dense_features/Time/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!dense_features/Time/strided_sliceStridedSlice"dense_features/Time/Shape:output:00dense_features/Time/strided_slice/stack:output:02dense_features/Time/strided_slice/stack_1:output:02dense_features/Time/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#dense_features/Time/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
!dense_features/Time/Reshape/shapePack*dense_features/Time/strided_slice:output:0,dense_features/Time/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/Time/ReshapeReshapeinputs_time*dense_features/Time/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????P
dense_features/V1/ShapeShape	inputs_v1*
T0*
_output_shapes
:o
%dense_features/V1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'dense_features/V1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'dense_features/V1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_features/V1/strided_sliceStridedSlice dense_features/V1/Shape:output:0.dense_features/V1/strided_slice/stack:output:00dense_features/V1/strided_slice/stack_1:output:00dense_features/V1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!dense_features/V1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dense_features/V1/Reshape/shapePack(dense_features/V1/strided_slice:output:0*dense_features/V1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V1/ReshapeReshape	inputs_v1(dense_features/V1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V10/ShapeShape
inputs_v10*
T0*
_output_shapes
:p
&dense_features/V10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V10/strided_sliceStridedSlice!dense_features/V10/Shape:output:0/dense_features/V10/strided_slice/stack:output:01dense_features/V10/strided_slice/stack_1:output:01dense_features/V10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V10/Reshape/shapePack)dense_features/V10/strided_slice:output:0+dense_features/V10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V10/ReshapeReshape
inputs_v10)dense_features/V10/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V11/ShapeShape
inputs_v11*
T0*
_output_shapes
:p
&dense_features/V11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V11/strided_sliceStridedSlice!dense_features/V11/Shape:output:0/dense_features/V11/strided_slice/stack:output:01dense_features/V11/strided_slice/stack_1:output:01dense_features/V11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V11/Reshape/shapePack)dense_features/V11/strided_slice:output:0+dense_features/V11/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V11/ReshapeReshape
inputs_v11)dense_features/V11/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V12/ShapeShape
inputs_v12*
T0*
_output_shapes
:p
&dense_features/V12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V12/strided_sliceStridedSlice!dense_features/V12/Shape:output:0/dense_features/V12/strided_slice/stack:output:01dense_features/V12/strided_slice/stack_1:output:01dense_features/V12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V12/Reshape/shapePack)dense_features/V12/strided_slice:output:0+dense_features/V12/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V12/ReshapeReshape
inputs_v12)dense_features/V12/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V13/ShapeShape
inputs_v13*
T0*
_output_shapes
:p
&dense_features/V13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V13/strided_sliceStridedSlice!dense_features/V13/Shape:output:0/dense_features/V13/strided_slice/stack:output:01dense_features/V13/strided_slice/stack_1:output:01dense_features/V13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V13/Reshape/shapePack)dense_features/V13/strided_slice:output:0+dense_features/V13/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V13/ReshapeReshape
inputs_v13)dense_features/V13/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V14/ShapeShape
inputs_v14*
T0*
_output_shapes
:p
&dense_features/V14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V14/strided_sliceStridedSlice!dense_features/V14/Shape:output:0/dense_features/V14/strided_slice/stack:output:01dense_features/V14/strided_slice/stack_1:output:01dense_features/V14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V14/Reshape/shapePack)dense_features/V14/strided_slice:output:0+dense_features/V14/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V14/ReshapeReshape
inputs_v14)dense_features/V14/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V15/ShapeShape
inputs_v15*
T0*
_output_shapes
:p
&dense_features/V15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V15/strided_sliceStridedSlice!dense_features/V15/Shape:output:0/dense_features/V15/strided_slice/stack:output:01dense_features/V15/strided_slice/stack_1:output:01dense_features/V15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V15/Reshape/shapePack)dense_features/V15/strided_slice:output:0+dense_features/V15/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V15/ReshapeReshape
inputs_v15)dense_features/V15/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V16/ShapeShape
inputs_v16*
T0*
_output_shapes
:p
&dense_features/V16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V16/strided_sliceStridedSlice!dense_features/V16/Shape:output:0/dense_features/V16/strided_slice/stack:output:01dense_features/V16/strided_slice/stack_1:output:01dense_features/V16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V16/Reshape/shapePack)dense_features/V16/strided_slice:output:0+dense_features/V16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V16/ReshapeReshape
inputs_v16)dense_features/V16/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V17/ShapeShape
inputs_v17*
T0*
_output_shapes
:p
&dense_features/V17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V17/strided_sliceStridedSlice!dense_features/V17/Shape:output:0/dense_features/V17/strided_slice/stack:output:01dense_features/V17/strided_slice/stack_1:output:01dense_features/V17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V17/Reshape/shapePack)dense_features/V17/strided_slice:output:0+dense_features/V17/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V17/ReshapeReshape
inputs_v17)dense_features/V17/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V18/ShapeShape
inputs_v18*
T0*
_output_shapes
:p
&dense_features/V18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V18/strided_sliceStridedSlice!dense_features/V18/Shape:output:0/dense_features/V18/strided_slice/stack:output:01dense_features/V18/strided_slice/stack_1:output:01dense_features/V18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V18/Reshape/shapePack)dense_features/V18/strided_slice:output:0+dense_features/V18/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V18/ReshapeReshape
inputs_v18)dense_features/V18/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V19/ShapeShape
inputs_v19*
T0*
_output_shapes
:p
&dense_features/V19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V19/strided_sliceStridedSlice!dense_features/V19/Shape:output:0/dense_features/V19/strided_slice/stack:output:01dense_features/V19/strided_slice/stack_1:output:01dense_features/V19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V19/Reshape/shapePack)dense_features/V19/strided_slice:output:0+dense_features/V19/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V19/ReshapeReshape
inputs_v19)dense_features/V19/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????P
dense_features/V2/ShapeShape	inputs_v2*
T0*
_output_shapes
:o
%dense_features/V2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'dense_features/V2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'dense_features/V2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_features/V2/strided_sliceStridedSlice dense_features/V2/Shape:output:0.dense_features/V2/strided_slice/stack:output:00dense_features/V2/strided_slice/stack_1:output:00dense_features/V2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!dense_features/V2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dense_features/V2/Reshape/shapePack(dense_features/V2/strided_slice:output:0*dense_features/V2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V2/ReshapeReshape	inputs_v2(dense_features/V2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V20/ShapeShape
inputs_v20*
T0*
_output_shapes
:p
&dense_features/V20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V20/strided_sliceStridedSlice!dense_features/V20/Shape:output:0/dense_features/V20/strided_slice/stack:output:01dense_features/V20/strided_slice/stack_1:output:01dense_features/V20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V20/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V20/Reshape/shapePack)dense_features/V20/strided_slice:output:0+dense_features/V20/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V20/ReshapeReshape
inputs_v20)dense_features/V20/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V21/ShapeShape
inputs_v21*
T0*
_output_shapes
:p
&dense_features/V21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V21/strided_sliceStridedSlice!dense_features/V21/Shape:output:0/dense_features/V21/strided_slice/stack:output:01dense_features/V21/strided_slice/stack_1:output:01dense_features/V21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V21/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V21/Reshape/shapePack)dense_features/V21/strided_slice:output:0+dense_features/V21/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V21/ReshapeReshape
inputs_v21)dense_features/V21/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V22/ShapeShape
inputs_v22*
T0*
_output_shapes
:p
&dense_features/V22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V22/strided_sliceStridedSlice!dense_features/V22/Shape:output:0/dense_features/V22/strided_slice/stack:output:01dense_features/V22/strided_slice/stack_1:output:01dense_features/V22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V22/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V22/Reshape/shapePack)dense_features/V22/strided_slice:output:0+dense_features/V22/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V22/ReshapeReshape
inputs_v22)dense_features/V22/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V23/ShapeShape
inputs_v23*
T0*
_output_shapes
:p
&dense_features/V23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V23/strided_sliceStridedSlice!dense_features/V23/Shape:output:0/dense_features/V23/strided_slice/stack:output:01dense_features/V23/strided_slice/stack_1:output:01dense_features/V23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V23/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V23/Reshape/shapePack)dense_features/V23/strided_slice:output:0+dense_features/V23/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V23/ReshapeReshape
inputs_v23)dense_features/V23/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V24/ShapeShape
inputs_v24*
T0*
_output_shapes
:p
&dense_features/V24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V24/strided_sliceStridedSlice!dense_features/V24/Shape:output:0/dense_features/V24/strided_slice/stack:output:01dense_features/V24/strided_slice/stack_1:output:01dense_features/V24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V24/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V24/Reshape/shapePack)dense_features/V24/strided_slice:output:0+dense_features/V24/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V24/ReshapeReshape
inputs_v24)dense_features/V24/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V25/ShapeShape
inputs_v25*
T0*
_output_shapes
:p
&dense_features/V25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V25/strided_sliceStridedSlice!dense_features/V25/Shape:output:0/dense_features/V25/strided_slice/stack:output:01dense_features/V25/strided_slice/stack_1:output:01dense_features/V25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V25/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V25/Reshape/shapePack)dense_features/V25/strided_slice:output:0+dense_features/V25/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V25/ReshapeReshape
inputs_v25)dense_features/V25/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V26/ShapeShape
inputs_v26*
T0*
_output_shapes
:p
&dense_features/V26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V26/strided_sliceStridedSlice!dense_features/V26/Shape:output:0/dense_features/V26/strided_slice/stack:output:01dense_features/V26/strided_slice/stack_1:output:01dense_features/V26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V26/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V26/Reshape/shapePack)dense_features/V26/strided_slice:output:0+dense_features/V26/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V26/ReshapeReshape
inputs_v26)dense_features/V26/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V27/ShapeShape
inputs_v27*
T0*
_output_shapes
:p
&dense_features/V27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V27/strided_sliceStridedSlice!dense_features/V27/Shape:output:0/dense_features/V27/strided_slice/stack:output:01dense_features/V27/strided_slice/stack_1:output:01dense_features/V27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V27/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V27/Reshape/shapePack)dense_features/V27/strided_slice:output:0+dense_features/V27/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V27/ReshapeReshape
inputs_v27)dense_features/V27/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V28/ShapeShape
inputs_v28*
T0*
_output_shapes
:p
&dense_features/V28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V28/strided_sliceStridedSlice!dense_features/V28/Shape:output:0/dense_features/V28/strided_slice/stack:output:01dense_features/V28/strided_slice/stack_1:output:01dense_features/V28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V28/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V28/Reshape/shapePack)dense_features/V28/strided_slice:output:0+dense_features/V28/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V28/ReshapeReshape
inputs_v28)dense_features/V28/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????P
dense_features/V3/ShapeShape	inputs_v3*
T0*
_output_shapes
:o
%dense_features/V3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'dense_features/V3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'dense_features/V3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_features/V3/strided_sliceStridedSlice dense_features/V3/Shape:output:0.dense_features/V3/strided_slice/stack:output:00dense_features/V3/strided_slice/stack_1:output:00dense_features/V3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!dense_features/V3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dense_features/V3/Reshape/shapePack(dense_features/V3/strided_slice:output:0*dense_features/V3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V3/ReshapeReshape	inputs_v3(dense_features/V3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????P
dense_features/V4/ShapeShape	inputs_v4*
T0*
_output_shapes
:o
%dense_features/V4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'dense_features/V4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'dense_features/V4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_features/V4/strided_sliceStridedSlice dense_features/V4/Shape:output:0.dense_features/V4/strided_slice/stack:output:00dense_features/V4/strided_slice/stack_1:output:00dense_features/V4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!dense_features/V4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dense_features/V4/Reshape/shapePack(dense_features/V4/strided_slice:output:0*dense_features/V4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V4/ReshapeReshape	inputs_v4(dense_features/V4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????P
dense_features/V5/ShapeShape	inputs_v5*
T0*
_output_shapes
:o
%dense_features/V5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'dense_features/V5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'dense_features/V5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_features/V5/strided_sliceStridedSlice dense_features/V5/Shape:output:0.dense_features/V5/strided_slice/stack:output:00dense_features/V5/strided_slice/stack_1:output:00dense_features/V5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!dense_features/V5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dense_features/V5/Reshape/shapePack(dense_features/V5/strided_slice:output:0*dense_features/V5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V5/ReshapeReshape	inputs_v5(dense_features/V5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????P
dense_features/V6/ShapeShape	inputs_v6*
T0*
_output_shapes
:o
%dense_features/V6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'dense_features/V6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'dense_features/V6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_features/V6/strided_sliceStridedSlice dense_features/V6/Shape:output:0.dense_features/V6/strided_slice/stack:output:00dense_features/V6/strided_slice/stack_1:output:00dense_features/V6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!dense_features/V6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dense_features/V6/Reshape/shapePack(dense_features/V6/strided_slice:output:0*dense_features/V6/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V6/ReshapeReshape	inputs_v6(dense_features/V6/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????P
dense_features/V7/ShapeShape	inputs_v7*
T0*
_output_shapes
:o
%dense_features/V7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'dense_features/V7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'dense_features/V7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_features/V7/strided_sliceStridedSlice dense_features/V7/Shape:output:0.dense_features/V7/strided_slice/stack:output:00dense_features/V7/strided_slice/stack_1:output:00dense_features/V7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!dense_features/V7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dense_features/V7/Reshape/shapePack(dense_features/V7/strided_slice:output:0*dense_features/V7/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V7/ReshapeReshape	inputs_v7(dense_features/V7/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????P
dense_features/V8/ShapeShape	inputs_v8*
T0*
_output_shapes
:o
%dense_features/V8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'dense_features/V8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'dense_features/V8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_features/V8/strided_sliceStridedSlice dense_features/V8/Shape:output:0.dense_features/V8/strided_slice/stack:output:00dense_features/V8/strided_slice/stack_1:output:00dense_features/V8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!dense_features/V8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dense_features/V8/Reshape/shapePack(dense_features/V8/strided_slice:output:0*dense_features/V8/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V8/ReshapeReshape	inputs_v8(dense_features/V8/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????P
dense_features/V9/ShapeShape	inputs_v9*
T0*
_output_shapes
:o
%dense_features/V9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'dense_features/V9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'dense_features/V9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_features/V9/strided_sliceStridedSlice dense_features/V9/Shape:output:0.dense_features/V9/strided_slice/stack:output:00dense_features/V9/strided_slice/stack_1:output:00dense_features/V9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!dense_features/V9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dense_features/V9/Reshape/shapePack(dense_features/V9/strided_slice:output:0*dense_features/V9/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V9/ReshapeReshape	inputs_v9(dense_features/V9/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????e
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????	
dense_features/concatConcatV2&dense_features/Amount/Reshape:output:0$dense_features/Time/Reshape:output:0"dense_features/V1/Reshape:output:0#dense_features/V10/Reshape:output:0#dense_features/V11/Reshape:output:0#dense_features/V12/Reshape:output:0#dense_features/V13/Reshape:output:0#dense_features/V14/Reshape:output:0#dense_features/V15/Reshape:output:0#dense_features/V16/Reshape:output:0#dense_features/V17/Reshape:output:0#dense_features/V18/Reshape:output:0#dense_features/V19/Reshape:output:0"dense_features/V2/Reshape:output:0#dense_features/V20/Reshape:output:0#dense_features/V21/Reshape:output:0#dense_features/V22/Reshape:output:0#dense_features/V23/Reshape:output:0#dense_features/V24/Reshape:output:0#dense_features/V25/Reshape:output:0#dense_features/V26/Reshape:output:0#dense_features/V27/Reshape:output:0#dense_features/V28/Reshape:output:0"dense_features/V3/Reshape:output:0"dense_features/V4/Reshape:output:0"dense_features/V5/Reshape:output:0"dense_features/V6/Reshape:output:0"dense_features/V7/Reshape:output:0"dense_features/V8/Reshape:output:0"dense_features/V9/Reshape:output:0#dense_features/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????|
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
 batch_normalization/moments/meanMeandense_features/concat:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(?
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes

:?
-batch_normalization/moments/SquaredDifferenceSquaredDifferencedense_features/concat:output:01batch_normalization/moments/StopGradient:output:0*
T0*'
_output_shapes
:??????????
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(?
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ?
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:?
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0p
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:?
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:?
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
#batch_normalization/batchnorm/mul_1Muldense_features/concat:output:0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:?
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:??????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:V R
'
_output_shapes
:?????????
'
_user_specified_nameinputs/Amount:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/Time:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V1:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V10:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V11:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V12:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V13:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V14:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V15:S	O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V16:S
O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V17:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V18:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V19:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V2:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V20:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V21:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V22:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V23:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V24:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V25:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V26:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V27:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V28:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V3:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V4:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V5:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V6:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V7:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V8:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V9
??
?
!__inference__wrapped_model_139468

amount
time
v1
v10
v11
v12
v13
v14
v15
v16
v17
v18
v19
v2
v20
v21
v22
v23
v24
v25
v26
v27
v28
v3
v4
v5
v6
v7
v8
v9I
;model_batch_normalization_batchnorm_readvariableop_resource:M
?model_batch_normalization_batchnorm_mul_readvariableop_resource:K
=model_batch_normalization_batchnorm_readvariableop_1_resource:K
=model_batch_normalization_batchnorm_readvariableop_2_resource:<
*model_dense_matmul_readvariableop_resource:9
+model_dense_biasadd_readvariableop_resource:
identity??2model/batch_normalization/batchnorm/ReadVariableOp?4model/batch_normalization/batchnorm/ReadVariableOp_1?4model/batch_normalization/batchnorm/ReadVariableOp_2?6model/batch_normalization/batchnorm/mul/ReadVariableOp?"model/dense/BiasAdd/ReadVariableOp?!model/dense/MatMul/ReadVariableOpW
!model/dense_features/Amount/ShapeShapeamount*
T0*
_output_shapes
:y
/model/dense_features/Amount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/dense_features/Amount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/dense_features/Amount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)model/dense_features/Amount/strided_sliceStridedSlice*model/dense_features/Amount/Shape:output:08model/dense_features/Amount/strided_slice/stack:output:0:model/dense_features/Amount/strided_slice/stack_1:output:0:model/dense_features/Amount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+model/dense_features/Amount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
)model/dense_features/Amount/Reshape/shapePack2model/dense_features/Amount/strided_slice:output:04model/dense_features/Amount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
#model/dense_features/Amount/ReshapeReshapeamount2model/dense_features/Amount/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????S
model/dense_features/Time/ShapeShapetime*
T0*
_output_shapes
:w
-model/dense_features/Time/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/model/dense_features/Time/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/model/dense_features/Time/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'model/dense_features/Time/strided_sliceStridedSlice(model/dense_features/Time/Shape:output:06model/dense_features/Time/strided_slice/stack:output:08model/dense_features/Time/strided_slice/stack_1:output:08model/dense_features/Time/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)model/dense_features/Time/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
'model/dense_features/Time/Reshape/shapePack0model/dense_features/Time/strided_slice:output:02model/dense_features/Time/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
!model/dense_features/Time/ReshapeReshapetime0model/dense_features/Time/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????O
model/dense_features/V1/ShapeShapev1*
T0*
_output_shapes
:u
+model/dense_features/V1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-model/dense_features/V1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-model/dense_features/V1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%model/dense_features/V1/strided_sliceStridedSlice&model/dense_features/V1/Shape:output:04model/dense_features/V1/strided_slice/stack:output:06model/dense_features/V1/strided_slice/stack_1:output:06model/dense_features/V1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'model/dense_features/V1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
%model/dense_features/V1/Reshape/shapePack.model/dense_features/V1/strided_slice:output:00model/dense_features/V1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
model/dense_features/V1/ReshapeReshapev1.model/dense_features/V1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????Q
model/dense_features/V10/ShapeShapev10*
T0*
_output_shapes
:v
,model/dense_features/V10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/dense_features/V10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/dense_features/V10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/dense_features/V10/strided_sliceStridedSlice'model/dense_features/V10/Shape:output:05model/dense_features/V10/strided_slice/stack:output:07model/dense_features/V10/strided_slice/stack_1:output:07model/dense_features/V10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/dense_features/V10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
&model/dense_features/V10/Reshape/shapePack/model/dense_features/V10/strided_slice:output:01model/dense_features/V10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
 model/dense_features/V10/ReshapeReshapev10/model/dense_features/V10/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????Q
model/dense_features/V11/ShapeShapev11*
T0*
_output_shapes
:v
,model/dense_features/V11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/dense_features/V11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/dense_features/V11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/dense_features/V11/strided_sliceStridedSlice'model/dense_features/V11/Shape:output:05model/dense_features/V11/strided_slice/stack:output:07model/dense_features/V11/strided_slice/stack_1:output:07model/dense_features/V11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/dense_features/V11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
&model/dense_features/V11/Reshape/shapePack/model/dense_features/V11/strided_slice:output:01model/dense_features/V11/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
 model/dense_features/V11/ReshapeReshapev11/model/dense_features/V11/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????Q
model/dense_features/V12/ShapeShapev12*
T0*
_output_shapes
:v
,model/dense_features/V12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/dense_features/V12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/dense_features/V12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/dense_features/V12/strided_sliceStridedSlice'model/dense_features/V12/Shape:output:05model/dense_features/V12/strided_slice/stack:output:07model/dense_features/V12/strided_slice/stack_1:output:07model/dense_features/V12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/dense_features/V12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
&model/dense_features/V12/Reshape/shapePack/model/dense_features/V12/strided_slice:output:01model/dense_features/V12/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
 model/dense_features/V12/ReshapeReshapev12/model/dense_features/V12/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????Q
model/dense_features/V13/ShapeShapev13*
T0*
_output_shapes
:v
,model/dense_features/V13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/dense_features/V13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/dense_features/V13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/dense_features/V13/strided_sliceStridedSlice'model/dense_features/V13/Shape:output:05model/dense_features/V13/strided_slice/stack:output:07model/dense_features/V13/strided_slice/stack_1:output:07model/dense_features/V13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/dense_features/V13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
&model/dense_features/V13/Reshape/shapePack/model/dense_features/V13/strided_slice:output:01model/dense_features/V13/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
 model/dense_features/V13/ReshapeReshapev13/model/dense_features/V13/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????Q
model/dense_features/V14/ShapeShapev14*
T0*
_output_shapes
:v
,model/dense_features/V14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/dense_features/V14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/dense_features/V14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/dense_features/V14/strided_sliceStridedSlice'model/dense_features/V14/Shape:output:05model/dense_features/V14/strided_slice/stack:output:07model/dense_features/V14/strided_slice/stack_1:output:07model/dense_features/V14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/dense_features/V14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
&model/dense_features/V14/Reshape/shapePack/model/dense_features/V14/strided_slice:output:01model/dense_features/V14/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
 model/dense_features/V14/ReshapeReshapev14/model/dense_features/V14/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????Q
model/dense_features/V15/ShapeShapev15*
T0*
_output_shapes
:v
,model/dense_features/V15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/dense_features/V15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/dense_features/V15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/dense_features/V15/strided_sliceStridedSlice'model/dense_features/V15/Shape:output:05model/dense_features/V15/strided_slice/stack:output:07model/dense_features/V15/strided_slice/stack_1:output:07model/dense_features/V15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/dense_features/V15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
&model/dense_features/V15/Reshape/shapePack/model/dense_features/V15/strided_slice:output:01model/dense_features/V15/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
 model/dense_features/V15/ReshapeReshapev15/model/dense_features/V15/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????Q
model/dense_features/V16/ShapeShapev16*
T0*
_output_shapes
:v
,model/dense_features/V16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/dense_features/V16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/dense_features/V16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/dense_features/V16/strided_sliceStridedSlice'model/dense_features/V16/Shape:output:05model/dense_features/V16/strided_slice/stack:output:07model/dense_features/V16/strided_slice/stack_1:output:07model/dense_features/V16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/dense_features/V16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
&model/dense_features/V16/Reshape/shapePack/model/dense_features/V16/strided_slice:output:01model/dense_features/V16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
 model/dense_features/V16/ReshapeReshapev16/model/dense_features/V16/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????Q
model/dense_features/V17/ShapeShapev17*
T0*
_output_shapes
:v
,model/dense_features/V17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/dense_features/V17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/dense_features/V17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/dense_features/V17/strided_sliceStridedSlice'model/dense_features/V17/Shape:output:05model/dense_features/V17/strided_slice/stack:output:07model/dense_features/V17/strided_slice/stack_1:output:07model/dense_features/V17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/dense_features/V17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
&model/dense_features/V17/Reshape/shapePack/model/dense_features/V17/strided_slice:output:01model/dense_features/V17/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
 model/dense_features/V17/ReshapeReshapev17/model/dense_features/V17/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????Q
model/dense_features/V18/ShapeShapev18*
T0*
_output_shapes
:v
,model/dense_features/V18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/dense_features/V18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/dense_features/V18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/dense_features/V18/strided_sliceStridedSlice'model/dense_features/V18/Shape:output:05model/dense_features/V18/strided_slice/stack:output:07model/dense_features/V18/strided_slice/stack_1:output:07model/dense_features/V18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/dense_features/V18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
&model/dense_features/V18/Reshape/shapePack/model/dense_features/V18/strided_slice:output:01model/dense_features/V18/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
 model/dense_features/V18/ReshapeReshapev18/model/dense_features/V18/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????Q
model/dense_features/V19/ShapeShapev19*
T0*
_output_shapes
:v
,model/dense_features/V19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/dense_features/V19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/dense_features/V19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/dense_features/V19/strided_sliceStridedSlice'model/dense_features/V19/Shape:output:05model/dense_features/V19/strided_slice/stack:output:07model/dense_features/V19/strided_slice/stack_1:output:07model/dense_features/V19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/dense_features/V19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
&model/dense_features/V19/Reshape/shapePack/model/dense_features/V19/strided_slice:output:01model/dense_features/V19/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
 model/dense_features/V19/ReshapeReshapev19/model/dense_features/V19/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????O
model/dense_features/V2/ShapeShapev2*
T0*
_output_shapes
:u
+model/dense_features/V2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-model/dense_features/V2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-model/dense_features/V2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%model/dense_features/V2/strided_sliceStridedSlice&model/dense_features/V2/Shape:output:04model/dense_features/V2/strided_slice/stack:output:06model/dense_features/V2/strided_slice/stack_1:output:06model/dense_features/V2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'model/dense_features/V2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
%model/dense_features/V2/Reshape/shapePack.model/dense_features/V2/strided_slice:output:00model/dense_features/V2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
model/dense_features/V2/ReshapeReshapev2.model/dense_features/V2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????Q
model/dense_features/V20/ShapeShapev20*
T0*
_output_shapes
:v
,model/dense_features/V20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/dense_features/V20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/dense_features/V20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/dense_features/V20/strided_sliceStridedSlice'model/dense_features/V20/Shape:output:05model/dense_features/V20/strided_slice/stack:output:07model/dense_features/V20/strided_slice/stack_1:output:07model/dense_features/V20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/dense_features/V20/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
&model/dense_features/V20/Reshape/shapePack/model/dense_features/V20/strided_slice:output:01model/dense_features/V20/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
 model/dense_features/V20/ReshapeReshapev20/model/dense_features/V20/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????Q
model/dense_features/V21/ShapeShapev21*
T0*
_output_shapes
:v
,model/dense_features/V21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/dense_features/V21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/dense_features/V21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/dense_features/V21/strided_sliceStridedSlice'model/dense_features/V21/Shape:output:05model/dense_features/V21/strided_slice/stack:output:07model/dense_features/V21/strided_slice/stack_1:output:07model/dense_features/V21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/dense_features/V21/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
&model/dense_features/V21/Reshape/shapePack/model/dense_features/V21/strided_slice:output:01model/dense_features/V21/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
 model/dense_features/V21/ReshapeReshapev21/model/dense_features/V21/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????Q
model/dense_features/V22/ShapeShapev22*
T0*
_output_shapes
:v
,model/dense_features/V22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/dense_features/V22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/dense_features/V22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/dense_features/V22/strided_sliceStridedSlice'model/dense_features/V22/Shape:output:05model/dense_features/V22/strided_slice/stack:output:07model/dense_features/V22/strided_slice/stack_1:output:07model/dense_features/V22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/dense_features/V22/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
&model/dense_features/V22/Reshape/shapePack/model/dense_features/V22/strided_slice:output:01model/dense_features/V22/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
 model/dense_features/V22/ReshapeReshapev22/model/dense_features/V22/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????Q
model/dense_features/V23/ShapeShapev23*
T0*
_output_shapes
:v
,model/dense_features/V23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/dense_features/V23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/dense_features/V23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/dense_features/V23/strided_sliceStridedSlice'model/dense_features/V23/Shape:output:05model/dense_features/V23/strided_slice/stack:output:07model/dense_features/V23/strided_slice/stack_1:output:07model/dense_features/V23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/dense_features/V23/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
&model/dense_features/V23/Reshape/shapePack/model/dense_features/V23/strided_slice:output:01model/dense_features/V23/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
 model/dense_features/V23/ReshapeReshapev23/model/dense_features/V23/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????Q
model/dense_features/V24/ShapeShapev24*
T0*
_output_shapes
:v
,model/dense_features/V24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/dense_features/V24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/dense_features/V24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/dense_features/V24/strided_sliceStridedSlice'model/dense_features/V24/Shape:output:05model/dense_features/V24/strided_slice/stack:output:07model/dense_features/V24/strided_slice/stack_1:output:07model/dense_features/V24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/dense_features/V24/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
&model/dense_features/V24/Reshape/shapePack/model/dense_features/V24/strided_slice:output:01model/dense_features/V24/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
 model/dense_features/V24/ReshapeReshapev24/model/dense_features/V24/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????Q
model/dense_features/V25/ShapeShapev25*
T0*
_output_shapes
:v
,model/dense_features/V25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/dense_features/V25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/dense_features/V25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/dense_features/V25/strided_sliceStridedSlice'model/dense_features/V25/Shape:output:05model/dense_features/V25/strided_slice/stack:output:07model/dense_features/V25/strided_slice/stack_1:output:07model/dense_features/V25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/dense_features/V25/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
&model/dense_features/V25/Reshape/shapePack/model/dense_features/V25/strided_slice:output:01model/dense_features/V25/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
 model/dense_features/V25/ReshapeReshapev25/model/dense_features/V25/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????Q
model/dense_features/V26/ShapeShapev26*
T0*
_output_shapes
:v
,model/dense_features/V26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/dense_features/V26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/dense_features/V26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/dense_features/V26/strided_sliceStridedSlice'model/dense_features/V26/Shape:output:05model/dense_features/V26/strided_slice/stack:output:07model/dense_features/V26/strided_slice/stack_1:output:07model/dense_features/V26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/dense_features/V26/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
&model/dense_features/V26/Reshape/shapePack/model/dense_features/V26/strided_slice:output:01model/dense_features/V26/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
 model/dense_features/V26/ReshapeReshapev26/model/dense_features/V26/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????Q
model/dense_features/V27/ShapeShapev27*
T0*
_output_shapes
:v
,model/dense_features/V27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/dense_features/V27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/dense_features/V27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/dense_features/V27/strided_sliceStridedSlice'model/dense_features/V27/Shape:output:05model/dense_features/V27/strided_slice/stack:output:07model/dense_features/V27/strided_slice/stack_1:output:07model/dense_features/V27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/dense_features/V27/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
&model/dense_features/V27/Reshape/shapePack/model/dense_features/V27/strided_slice:output:01model/dense_features/V27/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
 model/dense_features/V27/ReshapeReshapev27/model/dense_features/V27/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????Q
model/dense_features/V28/ShapeShapev28*
T0*
_output_shapes
:v
,model/dense_features/V28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/dense_features/V28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/dense_features/V28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/dense_features/V28/strided_sliceStridedSlice'model/dense_features/V28/Shape:output:05model/dense_features/V28/strided_slice/stack:output:07model/dense_features/V28/strided_slice/stack_1:output:07model/dense_features/V28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/dense_features/V28/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
&model/dense_features/V28/Reshape/shapePack/model/dense_features/V28/strided_slice:output:01model/dense_features/V28/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
 model/dense_features/V28/ReshapeReshapev28/model/dense_features/V28/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????O
model/dense_features/V3/ShapeShapev3*
T0*
_output_shapes
:u
+model/dense_features/V3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-model/dense_features/V3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-model/dense_features/V3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%model/dense_features/V3/strided_sliceStridedSlice&model/dense_features/V3/Shape:output:04model/dense_features/V3/strided_slice/stack:output:06model/dense_features/V3/strided_slice/stack_1:output:06model/dense_features/V3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'model/dense_features/V3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
%model/dense_features/V3/Reshape/shapePack.model/dense_features/V3/strided_slice:output:00model/dense_features/V3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
model/dense_features/V3/ReshapeReshapev3.model/dense_features/V3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????O
model/dense_features/V4/ShapeShapev4*
T0*
_output_shapes
:u
+model/dense_features/V4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-model/dense_features/V4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-model/dense_features/V4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%model/dense_features/V4/strided_sliceStridedSlice&model/dense_features/V4/Shape:output:04model/dense_features/V4/strided_slice/stack:output:06model/dense_features/V4/strided_slice/stack_1:output:06model/dense_features/V4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'model/dense_features/V4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
%model/dense_features/V4/Reshape/shapePack.model/dense_features/V4/strided_slice:output:00model/dense_features/V4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
model/dense_features/V4/ReshapeReshapev4.model/dense_features/V4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????O
model/dense_features/V5/ShapeShapev5*
T0*
_output_shapes
:u
+model/dense_features/V5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-model/dense_features/V5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-model/dense_features/V5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%model/dense_features/V5/strided_sliceStridedSlice&model/dense_features/V5/Shape:output:04model/dense_features/V5/strided_slice/stack:output:06model/dense_features/V5/strided_slice/stack_1:output:06model/dense_features/V5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'model/dense_features/V5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
%model/dense_features/V5/Reshape/shapePack.model/dense_features/V5/strided_slice:output:00model/dense_features/V5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
model/dense_features/V5/ReshapeReshapev5.model/dense_features/V5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????O
model/dense_features/V6/ShapeShapev6*
T0*
_output_shapes
:u
+model/dense_features/V6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-model/dense_features/V6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-model/dense_features/V6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%model/dense_features/V6/strided_sliceStridedSlice&model/dense_features/V6/Shape:output:04model/dense_features/V6/strided_slice/stack:output:06model/dense_features/V6/strided_slice/stack_1:output:06model/dense_features/V6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'model/dense_features/V6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
%model/dense_features/V6/Reshape/shapePack.model/dense_features/V6/strided_slice:output:00model/dense_features/V6/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
model/dense_features/V6/ReshapeReshapev6.model/dense_features/V6/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????O
model/dense_features/V7/ShapeShapev7*
T0*
_output_shapes
:u
+model/dense_features/V7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-model/dense_features/V7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-model/dense_features/V7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%model/dense_features/V7/strided_sliceStridedSlice&model/dense_features/V7/Shape:output:04model/dense_features/V7/strided_slice/stack:output:06model/dense_features/V7/strided_slice/stack_1:output:06model/dense_features/V7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'model/dense_features/V7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
%model/dense_features/V7/Reshape/shapePack.model/dense_features/V7/strided_slice:output:00model/dense_features/V7/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
model/dense_features/V7/ReshapeReshapev7.model/dense_features/V7/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????O
model/dense_features/V8/ShapeShapev8*
T0*
_output_shapes
:u
+model/dense_features/V8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-model/dense_features/V8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-model/dense_features/V8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%model/dense_features/V8/strided_sliceStridedSlice&model/dense_features/V8/Shape:output:04model/dense_features/V8/strided_slice/stack:output:06model/dense_features/V8/strided_slice/stack_1:output:06model/dense_features/V8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'model/dense_features/V8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
%model/dense_features/V8/Reshape/shapePack.model/dense_features/V8/strided_slice:output:00model/dense_features/V8/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
model/dense_features/V8/ReshapeReshapev8.model/dense_features/V8/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????O
model/dense_features/V9/ShapeShapev9*
T0*
_output_shapes
:u
+model/dense_features/V9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-model/dense_features/V9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-model/dense_features/V9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%model/dense_features/V9/strided_sliceStridedSlice&model/dense_features/V9/Shape:output:04model/dense_features/V9/strided_slice/stack:output:06model/dense_features/V9/strided_slice/stack_1:output:06model/dense_features/V9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'model/dense_features/V9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
%model/dense_features/V9/Reshape/shapePack.model/dense_features/V9/strided_slice:output:00model/dense_features/V9/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
model/dense_features/V9/ReshapeReshapev9.model/dense_features/V9/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????k
 model/dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
model/dense_features/concatConcatV2,model/dense_features/Amount/Reshape:output:0*model/dense_features/Time/Reshape:output:0(model/dense_features/V1/Reshape:output:0)model/dense_features/V10/Reshape:output:0)model/dense_features/V11/Reshape:output:0)model/dense_features/V12/Reshape:output:0)model/dense_features/V13/Reshape:output:0)model/dense_features/V14/Reshape:output:0)model/dense_features/V15/Reshape:output:0)model/dense_features/V16/Reshape:output:0)model/dense_features/V17/Reshape:output:0)model/dense_features/V18/Reshape:output:0)model/dense_features/V19/Reshape:output:0(model/dense_features/V2/Reshape:output:0)model/dense_features/V20/Reshape:output:0)model/dense_features/V21/Reshape:output:0)model/dense_features/V22/Reshape:output:0)model/dense_features/V23/Reshape:output:0)model/dense_features/V24/Reshape:output:0)model/dense_features/V25/Reshape:output:0)model/dense_features/V26/Reshape:output:0)model/dense_features/V27/Reshape:output:0)model/dense_features/V28/Reshape:output:0(model/dense_features/V3/Reshape:output:0(model/dense_features/V4/Reshape:output:0(model/dense_features/V5/Reshape:output:0(model/dense_features/V6/Reshape:output:0(model/dense_features/V7/Reshape:output:0(model/dense_features/V8/Reshape:output:0(model/dense_features/V9/Reshape:output:0)model/dense_features/concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
2model/batch_normalization/batchnorm/ReadVariableOpReadVariableOp;model_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0n
)model/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
'model/batch_normalization/batchnorm/addAddV2:model/batch_normalization/batchnorm/ReadVariableOp:value:02model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:?
)model/batch_normalization/batchnorm/RsqrtRsqrt+model/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:?
6model/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?model_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
'model/batch_normalization/batchnorm/mulMul-model/batch_normalization/batchnorm/Rsqrt:y:0>model/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
)model/batch_normalization/batchnorm/mul_1Mul$model/dense_features/concat:output:0+model/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
4model/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0?
)model/batch_normalization/batchnorm/mul_2Mul<model/batch_normalization/batchnorm/ReadVariableOp_1:value:0+model/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:?
4model/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0?
'model/batch_normalization/batchnorm/subSub<model/batch_normalization/batchnorm/ReadVariableOp_2:value:0-model/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
)model/batch_normalization/batchnorm/add_1AddV2-model/batch_normalization/batchnorm/mul_1:z:0+model/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:??????????
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model/dense/MatMulMatMul-model/batch_normalization/batchnorm/add_1:z:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
model/dense/SoftmaxSoftmaxmodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????l
IdentityIdentitymodel/dense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp3^model/batch_normalization/batchnorm/ReadVariableOp5^model/batch_normalization/batchnorm/ReadVariableOp_15^model/batch_normalization/batchnorm/ReadVariableOp_27^model/batch_normalization/batchnorm/mul/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : 2h
2model/batch_normalization/batchnorm/ReadVariableOp2model/batch_normalization/batchnorm/ReadVariableOp2l
4model/batch_normalization/batchnorm/ReadVariableOp_14model/batch_normalization/batchnorm/ReadVariableOp_12l
4model/batch_normalization/batchnorm/ReadVariableOp_24model/batch_normalization/batchnorm/ReadVariableOp_22p
6model/batch_normalization/batchnorm/mul/ReadVariableOp6model/batch_normalization/batchnorm/mul/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameAmount:MI
'
_output_shapes
:?????????

_user_specified_nameTime:KG
'
_output_shapes
:?????????

_user_specified_nameV1:LH
'
_output_shapes
:?????????

_user_specified_nameV10:LH
'
_output_shapes
:?????????

_user_specified_nameV11:LH
'
_output_shapes
:?????????

_user_specified_nameV12:LH
'
_output_shapes
:?????????

_user_specified_nameV13:LH
'
_output_shapes
:?????????

_user_specified_nameV14:LH
'
_output_shapes
:?????????

_user_specified_nameV15:L	H
'
_output_shapes
:?????????

_user_specified_nameV16:L
H
'
_output_shapes
:?????????

_user_specified_nameV17:LH
'
_output_shapes
:?????????

_user_specified_nameV18:LH
'
_output_shapes
:?????????

_user_specified_nameV19:KG
'
_output_shapes
:?????????

_user_specified_nameV2:LH
'
_output_shapes
:?????????

_user_specified_nameV20:LH
'
_output_shapes
:?????????

_user_specified_nameV21:LH
'
_output_shapes
:?????????

_user_specified_nameV22:LH
'
_output_shapes
:?????????

_user_specified_nameV23:LH
'
_output_shapes
:?????????

_user_specified_nameV24:LH
'
_output_shapes
:?????????

_user_specified_nameV25:LH
'
_output_shapes
:?????????

_user_specified_nameV26:LH
'
_output_shapes
:?????????

_user_specified_nameV27:LH
'
_output_shapes
:?????????

_user_specified_nameV28:KG
'
_output_shapes
:?????????

_user_specified_nameV3:KG
'
_output_shapes
:?????????

_user_specified_nameV4:KG
'
_output_shapes
:?????????

_user_specified_nameV5:KG
'
_output_shapes
:?????????

_user_specified_nameV6:KG
'
_output_shapes
:?????????

_user_specified_nameV7:KG
'
_output_shapes
:?????????

_user_specified_nameV8:KG
'
_output_shapes
:?????????

_user_specified_nameV9
?.
?
A__inference_model_layer_call_and_return_conditional_losses_139919

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17
	inputs_18
	inputs_19
	inputs_20
	inputs_21
	inputs_22
	inputs_23
	inputs_24
	inputs_25
	inputs_26
	inputs_27
	inputs_28
	inputs_29(
batch_normalization_139892:(
batch_normalization_139894:(
batch_normalization_139896:(
batch_normalization_139898:
dense_139913:
dense_139915:
identity??+batch_normalization/StatefulPartitionedCall?dense/StatefulPartitionedCall?
dense_features/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16	inputs_17	inputs_18	inputs_19	inputs_20	inputs_21	inputs_22	inputs_23	inputs_24	inputs_25	inputs_26	inputs_27	inputs_28	inputs_29*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_139890?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0batch_normalization_139892batch_normalization_139894batch_normalization_139896batch_normalization_139898*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_139492?
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_139913dense_139915*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_139912u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O
K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?	
A__inference_model_layer_call_and_return_conditional_losses_141012
inputs_amount
inputs_time
	inputs_v1

inputs_v10

inputs_v11

inputs_v12

inputs_v13

inputs_v14

inputs_v15

inputs_v16

inputs_v17

inputs_v18

inputs_v19
	inputs_v2

inputs_v20

inputs_v21

inputs_v22

inputs_v23

inputs_v24

inputs_v25

inputs_v26

inputs_v27

inputs_v28
	inputs_v3
	inputs_v4
	inputs_v5
	inputs_v6
	inputs_v7
	inputs_v8
	inputs_v9C
5batch_normalization_batchnorm_readvariableop_resource:G
9batch_normalization_batchnorm_mul_readvariableop_resource:E
7batch_normalization_batchnorm_readvariableop_1_resource:E
7batch_normalization_batchnorm_readvariableop_2_resource:6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity??,batch_normalization/batchnorm/ReadVariableOp?.batch_normalization/batchnorm/ReadVariableOp_1?.batch_normalization/batchnorm/ReadVariableOp_2?0batch_normalization/batchnorm/mul/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOpX
dense_features/Amount/ShapeShapeinputs_amount*
T0*
_output_shapes
:s
)dense_features/Amount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+dense_features/Amount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+dense_features/Amount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#dense_features/Amount/strided_sliceStridedSlice$dense_features/Amount/Shape:output:02dense_features/Amount/strided_slice/stack:output:04dense_features/Amount/strided_slice/stack_1:output:04dense_features/Amount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%dense_features/Amount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#dense_features/Amount/Reshape/shapePack,dense_features/Amount/strided_slice:output:0.dense_features/Amount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/Amount/ReshapeReshapeinputs_amount,dense_features/Amount/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????T
dense_features/Time/ShapeShapeinputs_time*
T0*
_output_shapes
:q
'dense_features/Time/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)dense_features/Time/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)dense_features/Time/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!dense_features/Time/strided_sliceStridedSlice"dense_features/Time/Shape:output:00dense_features/Time/strided_slice/stack:output:02dense_features/Time/strided_slice/stack_1:output:02dense_features/Time/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#dense_features/Time/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
!dense_features/Time/Reshape/shapePack*dense_features/Time/strided_slice:output:0,dense_features/Time/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/Time/ReshapeReshapeinputs_time*dense_features/Time/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????P
dense_features/V1/ShapeShape	inputs_v1*
T0*
_output_shapes
:o
%dense_features/V1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'dense_features/V1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'dense_features/V1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_features/V1/strided_sliceStridedSlice dense_features/V1/Shape:output:0.dense_features/V1/strided_slice/stack:output:00dense_features/V1/strided_slice/stack_1:output:00dense_features/V1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!dense_features/V1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dense_features/V1/Reshape/shapePack(dense_features/V1/strided_slice:output:0*dense_features/V1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V1/ReshapeReshape	inputs_v1(dense_features/V1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V10/ShapeShape
inputs_v10*
T0*
_output_shapes
:p
&dense_features/V10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V10/strided_sliceStridedSlice!dense_features/V10/Shape:output:0/dense_features/V10/strided_slice/stack:output:01dense_features/V10/strided_slice/stack_1:output:01dense_features/V10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V10/Reshape/shapePack)dense_features/V10/strided_slice:output:0+dense_features/V10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V10/ReshapeReshape
inputs_v10)dense_features/V10/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V11/ShapeShape
inputs_v11*
T0*
_output_shapes
:p
&dense_features/V11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V11/strided_sliceStridedSlice!dense_features/V11/Shape:output:0/dense_features/V11/strided_slice/stack:output:01dense_features/V11/strided_slice/stack_1:output:01dense_features/V11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V11/Reshape/shapePack)dense_features/V11/strided_slice:output:0+dense_features/V11/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V11/ReshapeReshape
inputs_v11)dense_features/V11/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V12/ShapeShape
inputs_v12*
T0*
_output_shapes
:p
&dense_features/V12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V12/strided_sliceStridedSlice!dense_features/V12/Shape:output:0/dense_features/V12/strided_slice/stack:output:01dense_features/V12/strided_slice/stack_1:output:01dense_features/V12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V12/Reshape/shapePack)dense_features/V12/strided_slice:output:0+dense_features/V12/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V12/ReshapeReshape
inputs_v12)dense_features/V12/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V13/ShapeShape
inputs_v13*
T0*
_output_shapes
:p
&dense_features/V13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V13/strided_sliceStridedSlice!dense_features/V13/Shape:output:0/dense_features/V13/strided_slice/stack:output:01dense_features/V13/strided_slice/stack_1:output:01dense_features/V13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V13/Reshape/shapePack)dense_features/V13/strided_slice:output:0+dense_features/V13/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V13/ReshapeReshape
inputs_v13)dense_features/V13/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V14/ShapeShape
inputs_v14*
T0*
_output_shapes
:p
&dense_features/V14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V14/strided_sliceStridedSlice!dense_features/V14/Shape:output:0/dense_features/V14/strided_slice/stack:output:01dense_features/V14/strided_slice/stack_1:output:01dense_features/V14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V14/Reshape/shapePack)dense_features/V14/strided_slice:output:0+dense_features/V14/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V14/ReshapeReshape
inputs_v14)dense_features/V14/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V15/ShapeShape
inputs_v15*
T0*
_output_shapes
:p
&dense_features/V15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V15/strided_sliceStridedSlice!dense_features/V15/Shape:output:0/dense_features/V15/strided_slice/stack:output:01dense_features/V15/strided_slice/stack_1:output:01dense_features/V15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V15/Reshape/shapePack)dense_features/V15/strided_slice:output:0+dense_features/V15/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V15/ReshapeReshape
inputs_v15)dense_features/V15/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V16/ShapeShape
inputs_v16*
T0*
_output_shapes
:p
&dense_features/V16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V16/strided_sliceStridedSlice!dense_features/V16/Shape:output:0/dense_features/V16/strided_slice/stack:output:01dense_features/V16/strided_slice/stack_1:output:01dense_features/V16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V16/Reshape/shapePack)dense_features/V16/strided_slice:output:0+dense_features/V16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V16/ReshapeReshape
inputs_v16)dense_features/V16/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V17/ShapeShape
inputs_v17*
T0*
_output_shapes
:p
&dense_features/V17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V17/strided_sliceStridedSlice!dense_features/V17/Shape:output:0/dense_features/V17/strided_slice/stack:output:01dense_features/V17/strided_slice/stack_1:output:01dense_features/V17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V17/Reshape/shapePack)dense_features/V17/strided_slice:output:0+dense_features/V17/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V17/ReshapeReshape
inputs_v17)dense_features/V17/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V18/ShapeShape
inputs_v18*
T0*
_output_shapes
:p
&dense_features/V18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V18/strided_sliceStridedSlice!dense_features/V18/Shape:output:0/dense_features/V18/strided_slice/stack:output:01dense_features/V18/strided_slice/stack_1:output:01dense_features/V18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V18/Reshape/shapePack)dense_features/V18/strided_slice:output:0+dense_features/V18/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V18/ReshapeReshape
inputs_v18)dense_features/V18/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V19/ShapeShape
inputs_v19*
T0*
_output_shapes
:p
&dense_features/V19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V19/strided_sliceStridedSlice!dense_features/V19/Shape:output:0/dense_features/V19/strided_slice/stack:output:01dense_features/V19/strided_slice/stack_1:output:01dense_features/V19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V19/Reshape/shapePack)dense_features/V19/strided_slice:output:0+dense_features/V19/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V19/ReshapeReshape
inputs_v19)dense_features/V19/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????P
dense_features/V2/ShapeShape	inputs_v2*
T0*
_output_shapes
:o
%dense_features/V2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'dense_features/V2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'dense_features/V2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_features/V2/strided_sliceStridedSlice dense_features/V2/Shape:output:0.dense_features/V2/strided_slice/stack:output:00dense_features/V2/strided_slice/stack_1:output:00dense_features/V2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!dense_features/V2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dense_features/V2/Reshape/shapePack(dense_features/V2/strided_slice:output:0*dense_features/V2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V2/ReshapeReshape	inputs_v2(dense_features/V2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V20/ShapeShape
inputs_v20*
T0*
_output_shapes
:p
&dense_features/V20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V20/strided_sliceStridedSlice!dense_features/V20/Shape:output:0/dense_features/V20/strided_slice/stack:output:01dense_features/V20/strided_slice/stack_1:output:01dense_features/V20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V20/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V20/Reshape/shapePack)dense_features/V20/strided_slice:output:0+dense_features/V20/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V20/ReshapeReshape
inputs_v20)dense_features/V20/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V21/ShapeShape
inputs_v21*
T0*
_output_shapes
:p
&dense_features/V21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V21/strided_sliceStridedSlice!dense_features/V21/Shape:output:0/dense_features/V21/strided_slice/stack:output:01dense_features/V21/strided_slice/stack_1:output:01dense_features/V21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V21/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V21/Reshape/shapePack)dense_features/V21/strided_slice:output:0+dense_features/V21/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V21/ReshapeReshape
inputs_v21)dense_features/V21/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V22/ShapeShape
inputs_v22*
T0*
_output_shapes
:p
&dense_features/V22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V22/strided_sliceStridedSlice!dense_features/V22/Shape:output:0/dense_features/V22/strided_slice/stack:output:01dense_features/V22/strided_slice/stack_1:output:01dense_features/V22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V22/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V22/Reshape/shapePack)dense_features/V22/strided_slice:output:0+dense_features/V22/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V22/ReshapeReshape
inputs_v22)dense_features/V22/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V23/ShapeShape
inputs_v23*
T0*
_output_shapes
:p
&dense_features/V23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V23/strided_sliceStridedSlice!dense_features/V23/Shape:output:0/dense_features/V23/strided_slice/stack:output:01dense_features/V23/strided_slice/stack_1:output:01dense_features/V23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V23/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V23/Reshape/shapePack)dense_features/V23/strided_slice:output:0+dense_features/V23/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V23/ReshapeReshape
inputs_v23)dense_features/V23/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V24/ShapeShape
inputs_v24*
T0*
_output_shapes
:p
&dense_features/V24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V24/strided_sliceStridedSlice!dense_features/V24/Shape:output:0/dense_features/V24/strided_slice/stack:output:01dense_features/V24/strided_slice/stack_1:output:01dense_features/V24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V24/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V24/Reshape/shapePack)dense_features/V24/strided_slice:output:0+dense_features/V24/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V24/ReshapeReshape
inputs_v24)dense_features/V24/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V25/ShapeShape
inputs_v25*
T0*
_output_shapes
:p
&dense_features/V25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V25/strided_sliceStridedSlice!dense_features/V25/Shape:output:0/dense_features/V25/strided_slice/stack:output:01dense_features/V25/strided_slice/stack_1:output:01dense_features/V25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V25/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V25/Reshape/shapePack)dense_features/V25/strided_slice:output:0+dense_features/V25/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V25/ReshapeReshape
inputs_v25)dense_features/V25/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V26/ShapeShape
inputs_v26*
T0*
_output_shapes
:p
&dense_features/V26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V26/strided_sliceStridedSlice!dense_features/V26/Shape:output:0/dense_features/V26/strided_slice/stack:output:01dense_features/V26/strided_slice/stack_1:output:01dense_features/V26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V26/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V26/Reshape/shapePack)dense_features/V26/strided_slice:output:0+dense_features/V26/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V26/ReshapeReshape
inputs_v26)dense_features/V26/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V27/ShapeShape
inputs_v27*
T0*
_output_shapes
:p
&dense_features/V27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V27/strided_sliceStridedSlice!dense_features/V27/Shape:output:0/dense_features/V27/strided_slice/stack:output:01dense_features/V27/strided_slice/stack_1:output:01dense_features/V27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V27/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V27/Reshape/shapePack)dense_features/V27/strided_slice:output:0+dense_features/V27/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V27/ReshapeReshape
inputs_v27)dense_features/V27/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????R
dense_features/V28/ShapeShape
inputs_v28*
T0*
_output_shapes
:p
&dense_features/V28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/V28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/V28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 dense_features/V28/strided_sliceStridedSlice!dense_features/V28/Shape:output:0/dense_features/V28/strided_slice/stack:output:01dense_features/V28/strided_slice/stack_1:output:01dense_features/V28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/V28/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 dense_features/V28/Reshape/shapePack)dense_features/V28/strided_slice:output:0+dense_features/V28/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V28/ReshapeReshape
inputs_v28)dense_features/V28/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????P
dense_features/V3/ShapeShape	inputs_v3*
T0*
_output_shapes
:o
%dense_features/V3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'dense_features/V3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'dense_features/V3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_features/V3/strided_sliceStridedSlice dense_features/V3/Shape:output:0.dense_features/V3/strided_slice/stack:output:00dense_features/V3/strided_slice/stack_1:output:00dense_features/V3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!dense_features/V3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dense_features/V3/Reshape/shapePack(dense_features/V3/strided_slice:output:0*dense_features/V3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V3/ReshapeReshape	inputs_v3(dense_features/V3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????P
dense_features/V4/ShapeShape	inputs_v4*
T0*
_output_shapes
:o
%dense_features/V4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'dense_features/V4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'dense_features/V4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_features/V4/strided_sliceStridedSlice dense_features/V4/Shape:output:0.dense_features/V4/strided_slice/stack:output:00dense_features/V4/strided_slice/stack_1:output:00dense_features/V4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!dense_features/V4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dense_features/V4/Reshape/shapePack(dense_features/V4/strided_slice:output:0*dense_features/V4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V4/ReshapeReshape	inputs_v4(dense_features/V4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????P
dense_features/V5/ShapeShape	inputs_v5*
T0*
_output_shapes
:o
%dense_features/V5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'dense_features/V5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'dense_features/V5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_features/V5/strided_sliceStridedSlice dense_features/V5/Shape:output:0.dense_features/V5/strided_slice/stack:output:00dense_features/V5/strided_slice/stack_1:output:00dense_features/V5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!dense_features/V5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dense_features/V5/Reshape/shapePack(dense_features/V5/strided_slice:output:0*dense_features/V5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V5/ReshapeReshape	inputs_v5(dense_features/V5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????P
dense_features/V6/ShapeShape	inputs_v6*
T0*
_output_shapes
:o
%dense_features/V6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'dense_features/V6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'dense_features/V6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_features/V6/strided_sliceStridedSlice dense_features/V6/Shape:output:0.dense_features/V6/strided_slice/stack:output:00dense_features/V6/strided_slice/stack_1:output:00dense_features/V6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!dense_features/V6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dense_features/V6/Reshape/shapePack(dense_features/V6/strided_slice:output:0*dense_features/V6/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V6/ReshapeReshape	inputs_v6(dense_features/V6/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????P
dense_features/V7/ShapeShape	inputs_v7*
T0*
_output_shapes
:o
%dense_features/V7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'dense_features/V7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'dense_features/V7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_features/V7/strided_sliceStridedSlice dense_features/V7/Shape:output:0.dense_features/V7/strided_slice/stack:output:00dense_features/V7/strided_slice/stack_1:output:00dense_features/V7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!dense_features/V7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dense_features/V7/Reshape/shapePack(dense_features/V7/strided_slice:output:0*dense_features/V7/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V7/ReshapeReshape	inputs_v7(dense_features/V7/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????P
dense_features/V8/ShapeShape	inputs_v8*
T0*
_output_shapes
:o
%dense_features/V8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'dense_features/V8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'dense_features/V8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_features/V8/strided_sliceStridedSlice dense_features/V8/Shape:output:0.dense_features/V8/strided_slice/stack:output:00dense_features/V8/strided_slice/stack_1:output:00dense_features/V8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!dense_features/V8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dense_features/V8/Reshape/shapePack(dense_features/V8/strided_slice:output:0*dense_features/V8/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V8/ReshapeReshape	inputs_v8(dense_features/V8/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????P
dense_features/V9/ShapeShape	inputs_v9*
T0*
_output_shapes
:o
%dense_features/V9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'dense_features/V9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'dense_features/V9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_features/V9/strided_sliceStridedSlice dense_features/V9/Shape:output:0.dense_features/V9/strided_slice/stack:output:00dense_features/V9/strided_slice/stack_1:output:00dense_features/V9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!dense_features/V9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dense_features/V9/Reshape/shapePack(dense_features/V9/strided_slice:output:0*dense_features/V9/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/V9/ReshapeReshape	inputs_v9(dense_features/V9/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????e
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????	
dense_features/concatConcatV2&dense_features/Amount/Reshape:output:0$dense_features/Time/Reshape:output:0"dense_features/V1/Reshape:output:0#dense_features/V10/Reshape:output:0#dense_features/V11/Reshape:output:0#dense_features/V12/Reshape:output:0#dense_features/V13/Reshape:output:0#dense_features/V14/Reshape:output:0#dense_features/V15/Reshape:output:0#dense_features/V16/Reshape:output:0#dense_features/V17/Reshape:output:0#dense_features/V18/Reshape:output:0#dense_features/V19/Reshape:output:0"dense_features/V2/Reshape:output:0#dense_features/V20/Reshape:output:0#dense_features/V21/Reshape:output:0#dense_features/V22/Reshape:output:0#dense_features/V23/Reshape:output:0#dense_features/V24/Reshape:output:0#dense_features/V25/Reshape:output:0#dense_features/V26/Reshape:output:0#dense_features/V27/Reshape:output:0#dense_features/V28/Reshape:output:0"dense_features/V3/Reshape:output:0"dense_features/V4/Reshape:output:0"dense_features/V5/Reshape:output:0"dense_features/V6/Reshape:output:0"dense_features/V7/Reshape:output:0"dense_features/V8/Reshape:output:0"dense_features/V9/Reshape:output:0#dense_features/concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:?
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
#batch_normalization/batchnorm/mul_1Muldense_features/concat:output:0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0?
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:?
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0?
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:??????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : 2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:V R
'
_output_shapes
:?????????
'
_user_specified_nameinputs/Amount:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/Time:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V1:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V10:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V11:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V12:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V13:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V14:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V15:S	O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V16:S
O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V17:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V18:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V19:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V2:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V20:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V21:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V22:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V23:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V24:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V25:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V26:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V27:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V28:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V3:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V4:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V5:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V6:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V7:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V8:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V9
?+
?
A__inference_model_layer_call_and_return_conditional_losses_140522

amount
time
v1
v10
v11
v12
v13
v14
v15
v16
v17
v18
v19
v2
v20
v21
v22
v23
v24
v25
v26
v27
v28
v3
v4
v5
v6
v7
v8
v9(
batch_normalization_140507:(
batch_normalization_140509:(
batch_normalization_140511:(
batch_normalization_140513:
dense_140516:
dense_140518:
identity??+batch_normalization/StatefulPartitionedCall?dense/StatefulPartitionedCall?
dense_features/PartitionedCallPartitionedCallamounttimev1v10v11v12v13v14v15v16v17v18v19v2v20v21v22v23v24v25v26v27v28v3v4v5v6v7v8v9*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_139890?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0batch_normalization_140507batch_normalization_140509batch_normalization_140511batch_normalization_140513*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_139492?
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_140516dense_140518*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_139912u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameAmount:MI
'
_output_shapes
:?????????

_user_specified_nameTime:KG
'
_output_shapes
:?????????

_user_specified_nameV1:LH
'
_output_shapes
:?????????

_user_specified_nameV10:LH
'
_output_shapes
:?????????

_user_specified_nameV11:LH
'
_output_shapes
:?????????

_user_specified_nameV12:LH
'
_output_shapes
:?????????

_user_specified_nameV13:LH
'
_output_shapes
:?????????

_user_specified_nameV14:LH
'
_output_shapes
:?????????

_user_specified_nameV15:L	H
'
_output_shapes
:?????????

_user_specified_nameV16:L
H
'
_output_shapes
:?????????

_user_specified_nameV17:LH
'
_output_shapes
:?????????

_user_specified_nameV18:LH
'
_output_shapes
:?????????

_user_specified_nameV19:KG
'
_output_shapes
:?????????

_user_specified_nameV2:LH
'
_output_shapes
:?????????

_user_specified_nameV20:LH
'
_output_shapes
:?????????

_user_specified_nameV21:LH
'
_output_shapes
:?????????

_user_specified_nameV22:LH
'
_output_shapes
:?????????

_user_specified_nameV23:LH
'
_output_shapes
:?????????

_user_specified_nameV24:LH
'
_output_shapes
:?????????

_user_specified_nameV25:LH
'
_output_shapes
:?????????

_user_specified_nameV26:LH
'
_output_shapes
:?????????

_user_specified_nameV27:LH
'
_output_shapes
:?????????

_user_specified_nameV28:KG
'
_output_shapes
:?????????

_user_specified_nameV3:KG
'
_output_shapes
:?????????

_user_specified_nameV4:KG
'
_output_shapes
:?????????

_user_specified_nameV5:KG
'
_output_shapes
:?????????

_user_specified_nameV6:KG
'
_output_shapes
:?????????

_user_specified_nameV7:KG
'
_output_shapes
:?????????

_user_specified_nameV8:KG
'
_output_shapes
:?????????

_user_specified_nameV9
?*
?
A__inference_model_layer_call_and_return_conditional_losses_140570

amount
time
v1
v10
v11
v12
v13
v14
v15
v16
v17
v18
v19
v2
v20
v21
v22
v23
v24
v25
v26
v27
v28
v3
v4
v5
v6
v7
v8
v9(
batch_normalization_140555:(
batch_normalization_140557:(
batch_normalization_140559:(
batch_normalization_140561:
dense_140564:
dense_140566:
identity??+batch_normalization/StatefulPartitionedCall?dense/StatefulPartitionedCall?
dense_features/PartitionedCallPartitionedCallamounttimev1v10v11v12v13v14v15v16v17v18v19v2v20v21v22v23v24v25v26v27v28v3v4v5v6v7v8v9*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_140285?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0batch_normalization_140555batch_normalization_140557batch_normalization_140559batch_normalization_140561*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_139539?
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_140564dense_140566*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_139912u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameAmount:MI
'
_output_shapes
:?????????

_user_specified_nameTime:KG
'
_output_shapes
:?????????

_user_specified_nameV1:LH
'
_output_shapes
:?????????

_user_specified_nameV10:LH
'
_output_shapes
:?????????

_user_specified_nameV11:LH
'
_output_shapes
:?????????

_user_specified_nameV12:LH
'
_output_shapes
:?????????

_user_specified_nameV13:LH
'
_output_shapes
:?????????

_user_specified_nameV14:LH
'
_output_shapes
:?????????

_user_specified_nameV15:L	H
'
_output_shapes
:?????????

_user_specified_nameV16:L
H
'
_output_shapes
:?????????

_user_specified_nameV17:LH
'
_output_shapes
:?????????

_user_specified_nameV18:LH
'
_output_shapes
:?????????

_user_specified_nameV19:KG
'
_output_shapes
:?????????

_user_specified_nameV2:LH
'
_output_shapes
:?????????

_user_specified_nameV20:LH
'
_output_shapes
:?????????

_user_specified_nameV21:LH
'
_output_shapes
:?????????

_user_specified_nameV22:LH
'
_output_shapes
:?????????

_user_specified_nameV23:LH
'
_output_shapes
:?????????

_user_specified_nameV24:LH
'
_output_shapes
:?????????

_user_specified_nameV25:LH
'
_output_shapes
:?????????

_user_specified_nameV26:LH
'
_output_shapes
:?????????

_user_specified_nameV27:LH
'
_output_shapes
:?????????

_user_specified_nameV28:KG
'
_output_shapes
:?????????

_user_specified_nameV3:KG
'
_output_shapes
:?????????

_user_specified_nameV4:KG
'
_output_shapes
:?????????

_user_specified_nameV5:KG
'
_output_shapes
:?????????

_user_specified_nameV6:KG
'
_output_shapes
:?????????

_user_specified_nameV7:KG
'
_output_shapes
:?????????

_user_specified_nameV8:KG
'
_output_shapes
:?????????

_user_specified_nameV9
??
?
J__inference_dense_features_layer_call_and_return_conditional_losses_140285
features

features_1

features_2

features_3

features_4

features_5

features_6

features_7

features_8

features_9
features_10
features_11
features_12
features_13
features_14
features_15
features_16
features_17
features_18
features_19
features_20
features_21
features_22
features_23
features_24
features_25
features_26
features_27
features_28
features_29
identityD
Amount/ShapeShapefeatures*
T0*
_output_shapes
:d
Amount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
Amount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
Amount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Amount/strided_sliceStridedSliceAmount/Shape:output:0#Amount/strided_slice/stack:output:0%Amount/strided_slice/stack_1:output:0%Amount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Amount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Amount/Reshape/shapePackAmount/strided_slice:output:0Amount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:t
Amount/ReshapeReshapefeaturesAmount/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D

Time/ShapeShape
features_1*
T0*
_output_shapes
:b
Time/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
Time/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
Time/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Time/strided_sliceStridedSliceTime/Shape:output:0!Time/strided_slice/stack:output:0#Time/strided_slice/stack_1:output:0#Time/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
Time/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Time/Reshape/shapePackTime/strided_slice:output:0Time/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
Time/ReshapeReshape
features_1Time/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????B
V1/ShapeShape
features_2*
T0*
_output_shapes
:`
V1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V1/strided_sliceStridedSliceV1/Shape:output:0V1/strided_slice/stack:output:0!V1/strided_slice/stack_1:output:0!V1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V1/Reshape/shapePackV1/strided_slice:output:0V1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:n

V1/ReshapeReshape
features_2V1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
	V10/ShapeShape
features_3*
T0*
_output_shapes
:a
V10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V10/strided_sliceStridedSliceV10/Shape:output:0 V10/strided_slice/stack:output:0"V10/strided_slice/stack_1:output:0"V10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V10/Reshape/shapePackV10/strided_slice:output:0V10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:p
V10/ReshapeReshape
features_3V10/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
	V11/ShapeShape
features_4*
T0*
_output_shapes
:a
V11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V11/strided_sliceStridedSliceV11/Shape:output:0 V11/strided_slice/stack:output:0"V11/strided_slice/stack_1:output:0"V11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V11/Reshape/shapePackV11/strided_slice:output:0V11/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:p
V11/ReshapeReshape
features_4V11/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
	V12/ShapeShape
features_5*
T0*
_output_shapes
:a
V12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V12/strided_sliceStridedSliceV12/Shape:output:0 V12/strided_slice/stack:output:0"V12/strided_slice/stack_1:output:0"V12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V12/Reshape/shapePackV12/strided_slice:output:0V12/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:p
V12/ReshapeReshape
features_5V12/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
	V13/ShapeShape
features_6*
T0*
_output_shapes
:a
V13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V13/strided_sliceStridedSliceV13/Shape:output:0 V13/strided_slice/stack:output:0"V13/strided_slice/stack_1:output:0"V13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V13/Reshape/shapePackV13/strided_slice:output:0V13/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:p
V13/ReshapeReshape
features_6V13/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
	V14/ShapeShape
features_7*
T0*
_output_shapes
:a
V14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V14/strided_sliceStridedSliceV14/Shape:output:0 V14/strided_slice/stack:output:0"V14/strided_slice/stack_1:output:0"V14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V14/Reshape/shapePackV14/strided_slice:output:0V14/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:p
V14/ReshapeReshape
features_7V14/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
	V15/ShapeShape
features_8*
T0*
_output_shapes
:a
V15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V15/strided_sliceStridedSliceV15/Shape:output:0 V15/strided_slice/stack:output:0"V15/strided_slice/stack_1:output:0"V15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V15/Reshape/shapePackV15/strided_slice:output:0V15/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:p
V15/ReshapeReshape
features_8V15/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
	V16/ShapeShape
features_9*
T0*
_output_shapes
:a
V16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V16/strided_sliceStridedSliceV16/Shape:output:0 V16/strided_slice/stack:output:0"V16/strided_slice/stack_1:output:0"V16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V16/Reshape/shapePackV16/strided_slice:output:0V16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:p
V16/ReshapeReshape
features_9V16/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V17/ShapeShapefeatures_10*
T0*
_output_shapes
:a
V17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V17/strided_sliceStridedSliceV17/Shape:output:0 V17/strided_slice/stack:output:0"V17/strided_slice/stack_1:output:0"V17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V17/Reshape/shapePackV17/strided_slice:output:0V17/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V17/ReshapeReshapefeatures_10V17/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V18/ShapeShapefeatures_11*
T0*
_output_shapes
:a
V18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V18/strided_sliceStridedSliceV18/Shape:output:0 V18/strided_slice/stack:output:0"V18/strided_slice/stack_1:output:0"V18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V18/Reshape/shapePackV18/strided_slice:output:0V18/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V18/ReshapeReshapefeatures_11V18/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V19/ShapeShapefeatures_12*
T0*
_output_shapes
:a
V19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V19/strided_sliceStridedSliceV19/Shape:output:0 V19/strided_slice/stack:output:0"V19/strided_slice/stack_1:output:0"V19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V19/Reshape/shapePackV19/strided_slice:output:0V19/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V19/ReshapeReshapefeatures_12V19/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V2/ShapeShapefeatures_13*
T0*
_output_shapes
:`
V2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V2/strided_sliceStridedSliceV2/Shape:output:0V2/strided_slice/stack:output:0!V2/strided_slice/stack_1:output:0!V2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V2/Reshape/shapePackV2/strided_slice:output:0V2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V2/ReshapeReshapefeatures_13V2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V20/ShapeShapefeatures_14*
T0*
_output_shapes
:a
V20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V20/strided_sliceStridedSliceV20/Shape:output:0 V20/strided_slice/stack:output:0"V20/strided_slice/stack_1:output:0"V20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V20/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V20/Reshape/shapePackV20/strided_slice:output:0V20/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V20/ReshapeReshapefeatures_14V20/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V21/ShapeShapefeatures_15*
T0*
_output_shapes
:a
V21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V21/strided_sliceStridedSliceV21/Shape:output:0 V21/strided_slice/stack:output:0"V21/strided_slice/stack_1:output:0"V21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V21/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V21/Reshape/shapePackV21/strided_slice:output:0V21/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V21/ReshapeReshapefeatures_15V21/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V22/ShapeShapefeatures_16*
T0*
_output_shapes
:a
V22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V22/strided_sliceStridedSliceV22/Shape:output:0 V22/strided_slice/stack:output:0"V22/strided_slice/stack_1:output:0"V22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V22/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V22/Reshape/shapePackV22/strided_slice:output:0V22/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V22/ReshapeReshapefeatures_16V22/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V23/ShapeShapefeatures_17*
T0*
_output_shapes
:a
V23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V23/strided_sliceStridedSliceV23/Shape:output:0 V23/strided_slice/stack:output:0"V23/strided_slice/stack_1:output:0"V23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V23/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V23/Reshape/shapePackV23/strided_slice:output:0V23/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V23/ReshapeReshapefeatures_17V23/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V24/ShapeShapefeatures_18*
T0*
_output_shapes
:a
V24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V24/strided_sliceStridedSliceV24/Shape:output:0 V24/strided_slice/stack:output:0"V24/strided_slice/stack_1:output:0"V24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V24/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V24/Reshape/shapePackV24/strided_slice:output:0V24/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V24/ReshapeReshapefeatures_18V24/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V25/ShapeShapefeatures_19*
T0*
_output_shapes
:a
V25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V25/strided_sliceStridedSliceV25/Shape:output:0 V25/strided_slice/stack:output:0"V25/strided_slice/stack_1:output:0"V25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V25/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V25/Reshape/shapePackV25/strided_slice:output:0V25/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V25/ReshapeReshapefeatures_19V25/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V26/ShapeShapefeatures_20*
T0*
_output_shapes
:a
V26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V26/strided_sliceStridedSliceV26/Shape:output:0 V26/strided_slice/stack:output:0"V26/strided_slice/stack_1:output:0"V26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V26/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V26/Reshape/shapePackV26/strided_slice:output:0V26/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V26/ReshapeReshapefeatures_20V26/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V27/ShapeShapefeatures_21*
T0*
_output_shapes
:a
V27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V27/strided_sliceStridedSliceV27/Shape:output:0 V27/strided_slice/stack:output:0"V27/strided_slice/stack_1:output:0"V27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V27/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V27/Reshape/shapePackV27/strided_slice:output:0V27/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V27/ReshapeReshapefeatures_21V27/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????D
	V28/ShapeShapefeatures_22*
T0*
_output_shapes
:a
V28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V28/strided_sliceStridedSliceV28/Shape:output:0 V28/strided_slice/stack:output:0"V28/strided_slice/stack_1:output:0"V28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V28/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V28/Reshape/shapePackV28/strided_slice:output:0V28/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:q
V28/ReshapeReshapefeatures_22V28/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V3/ShapeShapefeatures_23*
T0*
_output_shapes
:`
V3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V3/strided_sliceStridedSliceV3/Shape:output:0V3/strided_slice/stack:output:0!V3/strided_slice/stack_1:output:0!V3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V3/Reshape/shapePackV3/strided_slice:output:0V3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V3/ReshapeReshapefeatures_23V3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V4/ShapeShapefeatures_24*
T0*
_output_shapes
:`
V4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V4/strided_sliceStridedSliceV4/Shape:output:0V4/strided_slice/stack:output:0!V4/strided_slice/stack_1:output:0!V4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V4/Reshape/shapePackV4/strided_slice:output:0V4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V4/ReshapeReshapefeatures_24V4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V5/ShapeShapefeatures_25*
T0*
_output_shapes
:`
V5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V5/strided_sliceStridedSliceV5/Shape:output:0V5/strided_slice/stack:output:0!V5/strided_slice/stack_1:output:0!V5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V5/Reshape/shapePackV5/strided_slice:output:0V5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V5/ReshapeReshapefeatures_25V5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V6/ShapeShapefeatures_26*
T0*
_output_shapes
:`
V6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V6/strided_sliceStridedSliceV6/Shape:output:0V6/strided_slice/stack:output:0!V6/strided_slice/stack_1:output:0!V6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V6/Reshape/shapePackV6/strided_slice:output:0V6/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V6/ReshapeReshapefeatures_26V6/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V7/ShapeShapefeatures_27*
T0*
_output_shapes
:`
V7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V7/strided_sliceStridedSliceV7/Shape:output:0V7/strided_slice/stack:output:0!V7/strided_slice/stack_1:output:0!V7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V7/Reshape/shapePackV7/strided_slice:output:0V7/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V7/ReshapeReshapefeatures_27V7/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V8/ShapeShapefeatures_28*
T0*
_output_shapes
:`
V8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V8/strided_sliceStridedSliceV8/Shape:output:0V8/strided_slice/stack:output:0!V8/strided_slice/stack_1:output:0!V8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V8/Reshape/shapePackV8/strided_slice:output:0V8/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V8/ReshapeReshapefeatures_28V8/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V9/ShapeShapefeatures_29*
T0*
_output_shapes
:`
V9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V9/strided_sliceStridedSliceV9/Shape:output:0V9/strided_slice/stack:output:0!V9/strided_slice/stack_1:output:0!V9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V9/Reshape/shapePackV9/strided_slice:output:0V9/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V9/ReshapeReshapefeatures_29V9/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2Amount/Reshape:output:0Time/Reshape:output:0V1/Reshape:output:0V10/Reshape:output:0V11/Reshape:output:0V12/Reshape:output:0V13/Reshape:output:0V14/Reshape:output:0V15/Reshape:output:0V16/Reshape:output:0V17/Reshape:output:0V18/Reshape:output:0V19/Reshape:output:0V2/Reshape:output:0V20/Reshape:output:0V21/Reshape:output:0V22/Reshape:output:0V23/Reshape:output:0V24/Reshape:output:0V25/Reshape:output:0V26/Reshape:output:0V27/Reshape:output:0V28/Reshape:output:0V3/Reshape:output:0V4/Reshape:output:0V5/Reshape:output:0V6/Reshape:output:0V7/Reshape:output:0V8/Reshape:output:0V9/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
features:Q
M
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features
?
?
4__inference_batch_normalization_layer_call_fn_141968

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_139539o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?$
?
/__inference_dense_features_layer_call_fn_141392
features_amount
features_time
features_v1
features_v10
features_v11
features_v12
features_v13
features_v14
features_v15
features_v16
features_v17
features_v18
features_v19
features_v2
features_v20
features_v21
features_v22
features_v23
features_v24
features_v25
features_v26
features_v27
features_v28
features_v3
features_v4
features_v5
features_v6
features_v7
features_v8
features_v9
identity?
PartitionedCallPartitionedCallfeatures_amountfeatures_timefeatures_v1features_v10features_v11features_v12features_v13features_v14features_v15features_v16features_v17features_v18features_v19features_v2features_v20features_v21features_v22features_v23features_v24features_v25features_v26features_v27features_v28features_v3features_v4features_v5features_v6features_v7features_v8features_v9*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_140285`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:X T
'
_output_shapes
:?????????
)
_user_specified_namefeatures/Amount:VR
'
_output_shapes
:?????????
'
_user_specified_namefeatures/Time:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V1:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V10:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V11:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V12:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V13:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V14:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V15:U	Q
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V16:U
Q
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V17:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V18:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V19:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V2:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V20:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V21:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V22:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V23:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V24:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V25:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V26:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V27:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V28:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V3:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V4:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V5:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V6:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V7:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V8:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V9
?
?
&__inference_dense_layer_call_fn_142031

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_139912o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_142022

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?$
?
/__inference_dense_features_layer_call_fn_141358
features_amount
features_time
features_v1
features_v10
features_v11
features_v12
features_v13
features_v14
features_v15
features_v16
features_v17
features_v18
features_v19
features_v2
features_v20
features_v21
features_v22
features_v23
features_v24
features_v25
features_v26
features_v27
features_v28
features_v3
features_v4
features_v5
features_v6
features_v7
features_v8
features_v9
identity?
PartitionedCallPartitionedCallfeatures_amountfeatures_timefeatures_v1features_v10features_v11features_v12features_v13features_v14features_v15features_v16features_v17features_v18features_v19features_v2features_v20features_v21features_v22features_v23features_v24features_v25features_v26features_v27features_v28features_v3features_v4features_v5features_v6features_v7features_v8features_v9*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_139890`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:X T
'
_output_shapes
:?????????
)
_user_specified_namefeatures/Amount:VR
'
_output_shapes
:?????????
'
_user_specified_namefeatures/Time:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V1:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V10:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V11:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V12:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V13:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V14:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V15:U	Q
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V16:U
Q
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V17:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V18:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V19:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V2:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V20:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V21:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V22:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V23:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V24:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V25:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V26:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V27:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V28:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V3:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V4:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V5:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V6:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V7:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V8:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V9
??
?
J__inference_dense_features_layer_call_and_return_conditional_losses_141942
features_amount
features_time
features_v1
features_v10
features_v11
features_v12
features_v13
features_v14
features_v15
features_v16
features_v17
features_v18
features_v19
features_v2
features_v20
features_v21
features_v22
features_v23
features_v24
features_v25
features_v26
features_v27
features_v28
features_v3
features_v4
features_v5
features_v6
features_v7
features_v8
features_v9
identityK
Amount/ShapeShapefeatures_amount*
T0*
_output_shapes
:d
Amount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
Amount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
Amount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Amount/strided_sliceStridedSliceAmount/Shape:output:0#Amount/strided_slice/stack:output:0%Amount/strided_slice/stack_1:output:0%Amount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Amount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Amount/Reshape/shapePackAmount/strided_slice:output:0Amount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:{
Amount/ReshapeReshapefeatures_amountAmount/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????G

Time/ShapeShapefeatures_time*
T0*
_output_shapes
:b
Time/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
Time/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
Time/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Time/strided_sliceStridedSliceTime/Shape:output:0!Time/strided_slice/stack:output:0#Time/strided_slice/stack_1:output:0#Time/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
Time/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Time/Reshape/shapePackTime/strided_slice:output:0Time/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:u
Time/ReshapeReshapefeatures_timeTime/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V1/ShapeShapefeatures_v1*
T0*
_output_shapes
:`
V1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V1/strided_sliceStridedSliceV1/Shape:output:0V1/strided_slice/stack:output:0!V1/strided_slice/stack_1:output:0!V1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V1/Reshape/shapePackV1/strided_slice:output:0V1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V1/ReshapeReshapefeatures_v1V1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V10/ShapeShapefeatures_v10*
T0*
_output_shapes
:a
V10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V10/strided_sliceStridedSliceV10/Shape:output:0 V10/strided_slice/stack:output:0"V10/strided_slice/stack_1:output:0"V10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V10/Reshape/shapePackV10/strided_slice:output:0V10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V10/ReshapeReshapefeatures_v10V10/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V11/ShapeShapefeatures_v11*
T0*
_output_shapes
:a
V11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V11/strided_sliceStridedSliceV11/Shape:output:0 V11/strided_slice/stack:output:0"V11/strided_slice/stack_1:output:0"V11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V11/Reshape/shapePackV11/strided_slice:output:0V11/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V11/ReshapeReshapefeatures_v11V11/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V12/ShapeShapefeatures_v12*
T0*
_output_shapes
:a
V12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V12/strided_sliceStridedSliceV12/Shape:output:0 V12/strided_slice/stack:output:0"V12/strided_slice/stack_1:output:0"V12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V12/Reshape/shapePackV12/strided_slice:output:0V12/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V12/ReshapeReshapefeatures_v12V12/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V13/ShapeShapefeatures_v13*
T0*
_output_shapes
:a
V13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V13/strided_sliceStridedSliceV13/Shape:output:0 V13/strided_slice/stack:output:0"V13/strided_slice/stack_1:output:0"V13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V13/Reshape/shapePackV13/strided_slice:output:0V13/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V13/ReshapeReshapefeatures_v13V13/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V14/ShapeShapefeatures_v14*
T0*
_output_shapes
:a
V14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V14/strided_sliceStridedSliceV14/Shape:output:0 V14/strided_slice/stack:output:0"V14/strided_slice/stack_1:output:0"V14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V14/Reshape/shapePackV14/strided_slice:output:0V14/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V14/ReshapeReshapefeatures_v14V14/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V15/ShapeShapefeatures_v15*
T0*
_output_shapes
:a
V15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V15/strided_sliceStridedSliceV15/Shape:output:0 V15/strided_slice/stack:output:0"V15/strided_slice/stack_1:output:0"V15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V15/Reshape/shapePackV15/strided_slice:output:0V15/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V15/ReshapeReshapefeatures_v15V15/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V16/ShapeShapefeatures_v16*
T0*
_output_shapes
:a
V16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V16/strided_sliceStridedSliceV16/Shape:output:0 V16/strided_slice/stack:output:0"V16/strided_slice/stack_1:output:0"V16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V16/Reshape/shapePackV16/strided_slice:output:0V16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V16/ReshapeReshapefeatures_v16V16/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V17/ShapeShapefeatures_v17*
T0*
_output_shapes
:a
V17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V17/strided_sliceStridedSliceV17/Shape:output:0 V17/strided_slice/stack:output:0"V17/strided_slice/stack_1:output:0"V17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V17/Reshape/shapePackV17/strided_slice:output:0V17/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V17/ReshapeReshapefeatures_v17V17/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V18/ShapeShapefeatures_v18*
T0*
_output_shapes
:a
V18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V18/strided_sliceStridedSliceV18/Shape:output:0 V18/strided_slice/stack:output:0"V18/strided_slice/stack_1:output:0"V18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V18/Reshape/shapePackV18/strided_slice:output:0V18/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V18/ReshapeReshapefeatures_v18V18/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V19/ShapeShapefeatures_v19*
T0*
_output_shapes
:a
V19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V19/strided_sliceStridedSliceV19/Shape:output:0 V19/strided_slice/stack:output:0"V19/strided_slice/stack_1:output:0"V19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V19/Reshape/shapePackV19/strided_slice:output:0V19/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V19/ReshapeReshapefeatures_v19V19/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V2/ShapeShapefeatures_v2*
T0*
_output_shapes
:`
V2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V2/strided_sliceStridedSliceV2/Shape:output:0V2/strided_slice/stack:output:0!V2/strided_slice/stack_1:output:0!V2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V2/Reshape/shapePackV2/strided_slice:output:0V2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V2/ReshapeReshapefeatures_v2V2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V20/ShapeShapefeatures_v20*
T0*
_output_shapes
:a
V20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V20/strided_sliceStridedSliceV20/Shape:output:0 V20/strided_slice/stack:output:0"V20/strided_slice/stack_1:output:0"V20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V20/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V20/Reshape/shapePackV20/strided_slice:output:0V20/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V20/ReshapeReshapefeatures_v20V20/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V21/ShapeShapefeatures_v21*
T0*
_output_shapes
:a
V21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V21/strided_sliceStridedSliceV21/Shape:output:0 V21/strided_slice/stack:output:0"V21/strided_slice/stack_1:output:0"V21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V21/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V21/Reshape/shapePackV21/strided_slice:output:0V21/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V21/ReshapeReshapefeatures_v21V21/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V22/ShapeShapefeatures_v22*
T0*
_output_shapes
:a
V22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V22/strided_sliceStridedSliceV22/Shape:output:0 V22/strided_slice/stack:output:0"V22/strided_slice/stack_1:output:0"V22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V22/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V22/Reshape/shapePackV22/strided_slice:output:0V22/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V22/ReshapeReshapefeatures_v22V22/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V23/ShapeShapefeatures_v23*
T0*
_output_shapes
:a
V23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V23/strided_sliceStridedSliceV23/Shape:output:0 V23/strided_slice/stack:output:0"V23/strided_slice/stack_1:output:0"V23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V23/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V23/Reshape/shapePackV23/strided_slice:output:0V23/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V23/ReshapeReshapefeatures_v23V23/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V24/ShapeShapefeatures_v24*
T0*
_output_shapes
:a
V24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V24/strided_sliceStridedSliceV24/Shape:output:0 V24/strided_slice/stack:output:0"V24/strided_slice/stack_1:output:0"V24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V24/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V24/Reshape/shapePackV24/strided_slice:output:0V24/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V24/ReshapeReshapefeatures_v24V24/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V25/ShapeShapefeatures_v25*
T0*
_output_shapes
:a
V25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V25/strided_sliceStridedSliceV25/Shape:output:0 V25/strided_slice/stack:output:0"V25/strided_slice/stack_1:output:0"V25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V25/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V25/Reshape/shapePackV25/strided_slice:output:0V25/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V25/ReshapeReshapefeatures_v25V25/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V26/ShapeShapefeatures_v26*
T0*
_output_shapes
:a
V26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V26/strided_sliceStridedSliceV26/Shape:output:0 V26/strided_slice/stack:output:0"V26/strided_slice/stack_1:output:0"V26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V26/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V26/Reshape/shapePackV26/strided_slice:output:0V26/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V26/ReshapeReshapefeatures_v26V26/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V27/ShapeShapefeatures_v27*
T0*
_output_shapes
:a
V27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V27/strided_sliceStridedSliceV27/Shape:output:0 V27/strided_slice/stack:output:0"V27/strided_slice/stack_1:output:0"V27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V27/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V27/Reshape/shapePackV27/strided_slice:output:0V27/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V27/ReshapeReshapefeatures_v27V27/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V28/ShapeShapefeatures_v28*
T0*
_output_shapes
:a
V28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V28/strided_sliceStridedSliceV28/Shape:output:0 V28/strided_slice/stack:output:0"V28/strided_slice/stack_1:output:0"V28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V28/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V28/Reshape/shapePackV28/strided_slice:output:0V28/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V28/ReshapeReshapefeatures_v28V28/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V3/ShapeShapefeatures_v3*
T0*
_output_shapes
:`
V3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V3/strided_sliceStridedSliceV3/Shape:output:0V3/strided_slice/stack:output:0!V3/strided_slice/stack_1:output:0!V3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V3/Reshape/shapePackV3/strided_slice:output:0V3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V3/ReshapeReshapefeatures_v3V3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V4/ShapeShapefeatures_v4*
T0*
_output_shapes
:`
V4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V4/strided_sliceStridedSliceV4/Shape:output:0V4/strided_slice/stack:output:0!V4/strided_slice/stack_1:output:0!V4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V4/Reshape/shapePackV4/strided_slice:output:0V4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V4/ReshapeReshapefeatures_v4V4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V5/ShapeShapefeatures_v5*
T0*
_output_shapes
:`
V5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V5/strided_sliceStridedSliceV5/Shape:output:0V5/strided_slice/stack:output:0!V5/strided_slice/stack_1:output:0!V5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V5/Reshape/shapePackV5/strided_slice:output:0V5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V5/ReshapeReshapefeatures_v5V5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V6/ShapeShapefeatures_v6*
T0*
_output_shapes
:`
V6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V6/strided_sliceStridedSliceV6/Shape:output:0V6/strided_slice/stack:output:0!V6/strided_slice/stack_1:output:0!V6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V6/Reshape/shapePackV6/strided_slice:output:0V6/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V6/ReshapeReshapefeatures_v6V6/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V7/ShapeShapefeatures_v7*
T0*
_output_shapes
:`
V7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V7/strided_sliceStridedSliceV7/Shape:output:0V7/strided_slice/stack:output:0!V7/strided_slice/stack_1:output:0!V7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V7/Reshape/shapePackV7/strided_slice:output:0V7/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V7/ReshapeReshapefeatures_v7V7/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V8/ShapeShapefeatures_v8*
T0*
_output_shapes
:`
V8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V8/strided_sliceStridedSliceV8/Shape:output:0V8/strided_slice/stack:output:0!V8/strided_slice/stack_1:output:0!V8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V8/Reshape/shapePackV8/strided_slice:output:0V8/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V8/ReshapeReshapefeatures_v8V8/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V9/ShapeShapefeatures_v9*
T0*
_output_shapes
:`
V9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V9/strided_sliceStridedSliceV9/Shape:output:0V9/strided_slice/stack:output:0!V9/strided_slice/stack_1:output:0!V9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V9/Reshape/shapePackV9/strided_slice:output:0V9/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V9/ReshapeReshapefeatures_v9V9/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2Amount/Reshape:output:0Time/Reshape:output:0V1/Reshape:output:0V10/Reshape:output:0V11/Reshape:output:0V12/Reshape:output:0V13/Reshape:output:0V14/Reshape:output:0V15/Reshape:output:0V16/Reshape:output:0V17/Reshape:output:0V18/Reshape:output:0V19/Reshape:output:0V2/Reshape:output:0V20/Reshape:output:0V21/Reshape:output:0V22/Reshape:output:0V23/Reshape:output:0V24/Reshape:output:0V25/Reshape:output:0V26/Reshape:output:0V27/Reshape:output:0V28/Reshape:output:0V3/Reshape:output:0V4/Reshape:output:0V5/Reshape:output:0V6/Reshape:output:0V7/Reshape:output:0V8/Reshape:output:0V9/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:X T
'
_output_shapes
:?????????
)
_user_specified_namefeatures/Amount:VR
'
_output_shapes
:?????????
'
_user_specified_namefeatures/Time:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V1:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V10:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V11:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V12:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V13:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V14:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V15:U	Q
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V16:U
Q
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V17:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V18:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V19:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V2:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V20:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V21:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V22:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V23:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V24:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V25:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V26:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V27:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V28:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V3:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V4:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V5:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V6:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V7:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V8:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V9
?

?
A__inference_dense_layer_call_and_return_conditional_losses_139912

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
A__inference_dense_layer_call_and_return_conditional_losses_142042

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_141988

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
&__inference_model_layer_call_fn_140474

amount
time
v1
v10
v11
v12
v13
v14
v15
v16
v17
v18
v19
v2
v20
v21
v22
v23
v24
v25
v26
v27
v28
v3
v4
v5
v6
v7
v8
v9
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallamounttimev1v10v11v12v13v14v15v16v17v18v19v2v20v21v22v23v24v25v26v27v28v3v4v5v6v7v8v9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
 !"#*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_140413o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameAmount:MI
'
_output_shapes
:?????????

_user_specified_nameTime:KG
'
_output_shapes
:?????????

_user_specified_nameV1:LH
'
_output_shapes
:?????????

_user_specified_nameV10:LH
'
_output_shapes
:?????????

_user_specified_nameV11:LH
'
_output_shapes
:?????????

_user_specified_nameV12:LH
'
_output_shapes
:?????????

_user_specified_nameV13:LH
'
_output_shapes
:?????????

_user_specified_nameV14:LH
'
_output_shapes
:?????????

_user_specified_nameV15:L	H
'
_output_shapes
:?????????

_user_specified_nameV16:L
H
'
_output_shapes
:?????????

_user_specified_nameV17:LH
'
_output_shapes
:?????????

_user_specified_nameV18:LH
'
_output_shapes
:?????????

_user_specified_nameV19:KG
'
_output_shapes
:?????????

_user_specified_nameV2:LH
'
_output_shapes
:?????????

_user_specified_nameV20:LH
'
_output_shapes
:?????????

_user_specified_nameV21:LH
'
_output_shapes
:?????????

_user_specified_nameV22:LH
'
_output_shapes
:?????????

_user_specified_nameV23:LH
'
_output_shapes
:?????????

_user_specified_nameV24:LH
'
_output_shapes
:?????????

_user_specified_nameV25:LH
'
_output_shapes
:?????????

_user_specified_nameV26:LH
'
_output_shapes
:?????????

_user_specified_nameV27:LH
'
_output_shapes
:?????????

_user_specified_nameV28:KG
'
_output_shapes
:?????????

_user_specified_nameV3:KG
'
_output_shapes
:?????????

_user_specified_nameV4:KG
'
_output_shapes
:?????????

_user_specified_nameV5:KG
'
_output_shapes
:?????????

_user_specified_nameV6:KG
'
_output_shapes
:?????????

_user_specified_nameV7:KG
'
_output_shapes
:?????????

_user_specified_nameV8:KG
'
_output_shapes
:?????????

_user_specified_nameV9
?3
?	
__inference__traced_save_142160
file_prefix8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableopE
Asavev2_sgd_batch_normalization_gamma_momentum_read_readvariableopD
@savev2_sgd_batch_normalization_beta_momentum_read_readvariableop8
4savev2_sgd_dense_kernel_momentum_read_readvariableop6
2savev2_sgd_dense_bias_momentum_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?
B?
B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B ?	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableopAsavev2_sgd_batch_normalization_gamma_momentum_read_readvariableop@savev2_sgd_batch_normalization_beta_momentum_read_readvariableop4savev2_sgd_dense_kernel_momentum_read_readvariableop2savev2_sgd_dense_bias_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapesv
t: ::::::: : : : : : : : :?:?:?:?::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
?
?
4__inference_batch_normalization_layer_call_fn_141955

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_139492o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_139492

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?&
?
&__inference_model_layer_call_fn_140668
inputs_amount
inputs_time
	inputs_v1

inputs_v10

inputs_v11

inputs_v12

inputs_v13

inputs_v14

inputs_v15

inputs_v16

inputs_v17

inputs_v18

inputs_v19
	inputs_v2

inputs_v20

inputs_v21

inputs_v22

inputs_v23

inputs_v24

inputs_v25

inputs_v26

inputs_v27

inputs_v28
	inputs_v3
	inputs_v4
	inputs_v5
	inputs_v6
	inputs_v7
	inputs_v8
	inputs_v9
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_amountinputs_time	inputs_v1
inputs_v10
inputs_v11
inputs_v12
inputs_v13
inputs_v14
inputs_v15
inputs_v16
inputs_v17
inputs_v18
inputs_v19	inputs_v2
inputs_v20
inputs_v21
inputs_v22
inputs_v23
inputs_v24
inputs_v25
inputs_v26
inputs_v27
inputs_v28	inputs_v3	inputs_v4	inputs_v5	inputs_v6	inputs_v7	inputs_v8	inputs_v9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

 !"#*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_139919o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_nameinputs/Amount:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/Time:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V1:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V10:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V11:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V12:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V13:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V14:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V15:S	O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V16:S
O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V17:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V18:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V19:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V2:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V20:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V21:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V22:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V23:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V24:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V25:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V26:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V27:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/V28:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V3:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V4:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V5:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V6:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V7:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V8:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/V9
?.
?
A__inference_model_layer_call_and_return_conditional_losses_140413

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17
	inputs_18
	inputs_19
	inputs_20
	inputs_21
	inputs_22
	inputs_23
	inputs_24
	inputs_25
	inputs_26
	inputs_27
	inputs_28
	inputs_29(
batch_normalization_140398:(
batch_normalization_140400:(
batch_normalization_140402:(
batch_normalization_140404:
dense_140407:
dense_140409:
identity??+batch_normalization/StatefulPartitionedCall?dense/StatefulPartitionedCall?
dense_features/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16	inputs_17	inputs_18	inputs_19	inputs_20	inputs_21	inputs_22	inputs_23	inputs_24	inputs_25	inputs_26	inputs_27	inputs_28	inputs_29*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_140285?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0batch_normalization_140398batch_normalization_140400batch_normalization_140402batch_normalization_140404*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_139539?
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_140407dense_140409*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_139912u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O
K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
&__inference_model_layer_call_fn_139934

amount
time
v1
v10
v11
v12
v13
v14
v15
v16
v17
v18
v19
v2
v20
v21
v22
v23
v24
v25
v26
v27
v28
v3
v4
v5
v6
v7
v8
v9
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallamounttimev1v10v11v12v13v14v15v16v17v18v19v2v20v21v22v23v24v25v26v27v28v3v4v5v6v7v8v9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

 !"#*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_139919o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameAmount:MI
'
_output_shapes
:?????????

_user_specified_nameTime:KG
'
_output_shapes
:?????????

_user_specified_nameV1:LH
'
_output_shapes
:?????????

_user_specified_nameV10:LH
'
_output_shapes
:?????????

_user_specified_nameV11:LH
'
_output_shapes
:?????????

_user_specified_nameV12:LH
'
_output_shapes
:?????????

_user_specified_nameV13:LH
'
_output_shapes
:?????????

_user_specified_nameV14:LH
'
_output_shapes
:?????????

_user_specified_nameV15:L	H
'
_output_shapes
:?????????

_user_specified_nameV16:L
H
'
_output_shapes
:?????????

_user_specified_nameV17:LH
'
_output_shapes
:?????????

_user_specified_nameV18:LH
'
_output_shapes
:?????????

_user_specified_nameV19:KG
'
_output_shapes
:?????????

_user_specified_nameV2:LH
'
_output_shapes
:?????????

_user_specified_nameV20:LH
'
_output_shapes
:?????????

_user_specified_nameV21:LH
'
_output_shapes
:?????????

_user_specified_nameV22:LH
'
_output_shapes
:?????????

_user_specified_nameV23:LH
'
_output_shapes
:?????????

_user_specified_nameV24:LH
'
_output_shapes
:?????????

_user_specified_nameV25:LH
'
_output_shapes
:?????????

_user_specified_nameV26:LH
'
_output_shapes
:?????????

_user_specified_nameV27:LH
'
_output_shapes
:?????????

_user_specified_nameV28:KG
'
_output_shapes
:?????????

_user_specified_nameV3:KG
'
_output_shapes
:?????????

_user_specified_nameV4:KG
'
_output_shapes
:?????????

_user_specified_nameV5:KG
'
_output_shapes
:?????????

_user_specified_nameV6:KG
'
_output_shapes
:?????????

_user_specified_nameV7:KG
'
_output_shapes
:?????????

_user_specified_nameV8:KG
'
_output_shapes
:?????????

_user_specified_nameV9
?!
?
$__inference_signature_wrapper_140622

amount
time
v1
v10
v11
v12
v13
v14
v15
v16
v17
v18
v19
v2
v20
v21
v22
v23
v24
v25
v26
v27
v28
v3
v4
v5
v6
v7
v8
v9
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallamounttimev1v10v11v12v13v14v15v16v17v18v19v2v20v21v22v23v24v25v26v27v28v3v4v5v6v7v8v9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

 !"#*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_139468o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameAmount:MI
'
_output_shapes
:?????????

_user_specified_nameTime:KG
'
_output_shapes
:?????????

_user_specified_nameV1:LH
'
_output_shapes
:?????????

_user_specified_nameV10:LH
'
_output_shapes
:?????????

_user_specified_nameV11:LH
'
_output_shapes
:?????????

_user_specified_nameV12:LH
'
_output_shapes
:?????????

_user_specified_nameV13:LH
'
_output_shapes
:?????????

_user_specified_nameV14:LH
'
_output_shapes
:?????????

_user_specified_nameV15:L	H
'
_output_shapes
:?????????

_user_specified_nameV16:L
H
'
_output_shapes
:?????????

_user_specified_nameV17:LH
'
_output_shapes
:?????????

_user_specified_nameV18:LH
'
_output_shapes
:?????????

_user_specified_nameV19:KG
'
_output_shapes
:?????????

_user_specified_nameV2:LH
'
_output_shapes
:?????????

_user_specified_nameV20:LH
'
_output_shapes
:?????????

_user_specified_nameV21:LH
'
_output_shapes
:?????????

_user_specified_nameV22:LH
'
_output_shapes
:?????????

_user_specified_nameV23:LH
'
_output_shapes
:?????????

_user_specified_nameV24:LH
'
_output_shapes
:?????????

_user_specified_nameV25:LH
'
_output_shapes
:?????????

_user_specified_nameV26:LH
'
_output_shapes
:?????????

_user_specified_nameV27:LH
'
_output_shapes
:?????????

_user_specified_nameV28:KG
'
_output_shapes
:?????????

_user_specified_nameV3:KG
'
_output_shapes
:?????????

_user_specified_nameV4:KG
'
_output_shapes
:?????????

_user_specified_nameV5:KG
'
_output_shapes
:?????????

_user_specified_nameV6:KG
'
_output_shapes
:?????????

_user_specified_nameV7:KG
'
_output_shapes
:?????????

_user_specified_nameV8:KG
'
_output_shapes
:?????????

_user_specified_nameV9
??
?
J__inference_dense_features_layer_call_and_return_conditional_losses_141667
features_amount
features_time
features_v1
features_v10
features_v11
features_v12
features_v13
features_v14
features_v15
features_v16
features_v17
features_v18
features_v19
features_v2
features_v20
features_v21
features_v22
features_v23
features_v24
features_v25
features_v26
features_v27
features_v28
features_v3
features_v4
features_v5
features_v6
features_v7
features_v8
features_v9
identityK
Amount/ShapeShapefeatures_amount*
T0*
_output_shapes
:d
Amount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
Amount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
Amount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Amount/strided_sliceStridedSliceAmount/Shape:output:0#Amount/strided_slice/stack:output:0%Amount/strided_slice/stack_1:output:0%Amount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Amount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Amount/Reshape/shapePackAmount/strided_slice:output:0Amount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:{
Amount/ReshapeReshapefeatures_amountAmount/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????G

Time/ShapeShapefeatures_time*
T0*
_output_shapes
:b
Time/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
Time/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
Time/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Time/strided_sliceStridedSliceTime/Shape:output:0!Time/strided_slice/stack:output:0#Time/strided_slice/stack_1:output:0#Time/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
Time/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Time/Reshape/shapePackTime/strided_slice:output:0Time/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:u
Time/ReshapeReshapefeatures_timeTime/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V1/ShapeShapefeatures_v1*
T0*
_output_shapes
:`
V1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V1/strided_sliceStridedSliceV1/Shape:output:0V1/strided_slice/stack:output:0!V1/strided_slice/stack_1:output:0!V1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V1/Reshape/shapePackV1/strided_slice:output:0V1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V1/ReshapeReshapefeatures_v1V1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V10/ShapeShapefeatures_v10*
T0*
_output_shapes
:a
V10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V10/strided_sliceStridedSliceV10/Shape:output:0 V10/strided_slice/stack:output:0"V10/strided_slice/stack_1:output:0"V10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V10/Reshape/shapePackV10/strided_slice:output:0V10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V10/ReshapeReshapefeatures_v10V10/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V11/ShapeShapefeatures_v11*
T0*
_output_shapes
:a
V11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V11/strided_sliceStridedSliceV11/Shape:output:0 V11/strided_slice/stack:output:0"V11/strided_slice/stack_1:output:0"V11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V11/Reshape/shapePackV11/strided_slice:output:0V11/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V11/ReshapeReshapefeatures_v11V11/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V12/ShapeShapefeatures_v12*
T0*
_output_shapes
:a
V12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V12/strided_sliceStridedSliceV12/Shape:output:0 V12/strided_slice/stack:output:0"V12/strided_slice/stack_1:output:0"V12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V12/Reshape/shapePackV12/strided_slice:output:0V12/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V12/ReshapeReshapefeatures_v12V12/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V13/ShapeShapefeatures_v13*
T0*
_output_shapes
:a
V13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V13/strided_sliceStridedSliceV13/Shape:output:0 V13/strided_slice/stack:output:0"V13/strided_slice/stack_1:output:0"V13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V13/Reshape/shapePackV13/strided_slice:output:0V13/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V13/ReshapeReshapefeatures_v13V13/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V14/ShapeShapefeatures_v14*
T0*
_output_shapes
:a
V14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V14/strided_sliceStridedSliceV14/Shape:output:0 V14/strided_slice/stack:output:0"V14/strided_slice/stack_1:output:0"V14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V14/Reshape/shapePackV14/strided_slice:output:0V14/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V14/ReshapeReshapefeatures_v14V14/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V15/ShapeShapefeatures_v15*
T0*
_output_shapes
:a
V15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V15/strided_sliceStridedSliceV15/Shape:output:0 V15/strided_slice/stack:output:0"V15/strided_slice/stack_1:output:0"V15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V15/Reshape/shapePackV15/strided_slice:output:0V15/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V15/ReshapeReshapefeatures_v15V15/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V16/ShapeShapefeatures_v16*
T0*
_output_shapes
:a
V16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V16/strided_sliceStridedSliceV16/Shape:output:0 V16/strided_slice/stack:output:0"V16/strided_slice/stack_1:output:0"V16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V16/Reshape/shapePackV16/strided_slice:output:0V16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V16/ReshapeReshapefeatures_v16V16/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V17/ShapeShapefeatures_v17*
T0*
_output_shapes
:a
V17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V17/strided_sliceStridedSliceV17/Shape:output:0 V17/strided_slice/stack:output:0"V17/strided_slice/stack_1:output:0"V17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V17/Reshape/shapePackV17/strided_slice:output:0V17/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V17/ReshapeReshapefeatures_v17V17/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V18/ShapeShapefeatures_v18*
T0*
_output_shapes
:a
V18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V18/strided_sliceStridedSliceV18/Shape:output:0 V18/strided_slice/stack:output:0"V18/strided_slice/stack_1:output:0"V18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V18/Reshape/shapePackV18/strided_slice:output:0V18/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V18/ReshapeReshapefeatures_v18V18/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V19/ShapeShapefeatures_v19*
T0*
_output_shapes
:a
V19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V19/strided_sliceStridedSliceV19/Shape:output:0 V19/strided_slice/stack:output:0"V19/strided_slice/stack_1:output:0"V19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V19/Reshape/shapePackV19/strided_slice:output:0V19/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V19/ReshapeReshapefeatures_v19V19/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V2/ShapeShapefeatures_v2*
T0*
_output_shapes
:`
V2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V2/strided_sliceStridedSliceV2/Shape:output:0V2/strided_slice/stack:output:0!V2/strided_slice/stack_1:output:0!V2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V2/Reshape/shapePackV2/strided_slice:output:0V2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V2/ReshapeReshapefeatures_v2V2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V20/ShapeShapefeatures_v20*
T0*
_output_shapes
:a
V20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V20/strided_sliceStridedSliceV20/Shape:output:0 V20/strided_slice/stack:output:0"V20/strided_slice/stack_1:output:0"V20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V20/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V20/Reshape/shapePackV20/strided_slice:output:0V20/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V20/ReshapeReshapefeatures_v20V20/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V21/ShapeShapefeatures_v21*
T0*
_output_shapes
:a
V21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V21/strided_sliceStridedSliceV21/Shape:output:0 V21/strided_slice/stack:output:0"V21/strided_slice/stack_1:output:0"V21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V21/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V21/Reshape/shapePackV21/strided_slice:output:0V21/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V21/ReshapeReshapefeatures_v21V21/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V22/ShapeShapefeatures_v22*
T0*
_output_shapes
:a
V22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V22/strided_sliceStridedSliceV22/Shape:output:0 V22/strided_slice/stack:output:0"V22/strided_slice/stack_1:output:0"V22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V22/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V22/Reshape/shapePackV22/strided_slice:output:0V22/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V22/ReshapeReshapefeatures_v22V22/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V23/ShapeShapefeatures_v23*
T0*
_output_shapes
:a
V23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V23/strided_sliceStridedSliceV23/Shape:output:0 V23/strided_slice/stack:output:0"V23/strided_slice/stack_1:output:0"V23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V23/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V23/Reshape/shapePackV23/strided_slice:output:0V23/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V23/ReshapeReshapefeatures_v23V23/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V24/ShapeShapefeatures_v24*
T0*
_output_shapes
:a
V24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V24/strided_sliceStridedSliceV24/Shape:output:0 V24/strided_slice/stack:output:0"V24/strided_slice/stack_1:output:0"V24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V24/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V24/Reshape/shapePackV24/strided_slice:output:0V24/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V24/ReshapeReshapefeatures_v24V24/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V25/ShapeShapefeatures_v25*
T0*
_output_shapes
:a
V25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V25/strided_sliceStridedSliceV25/Shape:output:0 V25/strided_slice/stack:output:0"V25/strided_slice/stack_1:output:0"V25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V25/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V25/Reshape/shapePackV25/strided_slice:output:0V25/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V25/ReshapeReshapefeatures_v25V25/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V26/ShapeShapefeatures_v26*
T0*
_output_shapes
:a
V26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V26/strided_sliceStridedSliceV26/Shape:output:0 V26/strided_slice/stack:output:0"V26/strided_slice/stack_1:output:0"V26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V26/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V26/Reshape/shapePackV26/strided_slice:output:0V26/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V26/ReshapeReshapefeatures_v26V26/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V27/ShapeShapefeatures_v27*
T0*
_output_shapes
:a
V27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V27/strided_sliceStridedSliceV27/Shape:output:0 V27/strided_slice/stack:output:0"V27/strided_slice/stack_1:output:0"V27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V27/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V27/Reshape/shapePackV27/strided_slice:output:0V27/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V27/ReshapeReshapefeatures_v27V27/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E
	V28/ShapeShapefeatures_v28*
T0*
_output_shapes
:a
V28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
V28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
V28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V28/strided_sliceStridedSliceV28/Shape:output:0 V28/strided_slice/stack:output:0"V28/strided_slice/stack_1:output:0"V28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
V28/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
V28/Reshape/shapePackV28/strided_slice:output:0V28/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
V28/ReshapeReshapefeatures_v28V28/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V3/ShapeShapefeatures_v3*
T0*
_output_shapes
:`
V3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V3/strided_sliceStridedSliceV3/Shape:output:0V3/strided_slice/stack:output:0!V3/strided_slice/stack_1:output:0!V3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V3/Reshape/shapePackV3/strided_slice:output:0V3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V3/ReshapeReshapefeatures_v3V3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V4/ShapeShapefeatures_v4*
T0*
_output_shapes
:`
V4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V4/strided_sliceStridedSliceV4/Shape:output:0V4/strided_slice/stack:output:0!V4/strided_slice/stack_1:output:0!V4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V4/Reshape/shapePackV4/strided_slice:output:0V4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V4/ReshapeReshapefeatures_v4V4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V5/ShapeShapefeatures_v5*
T0*
_output_shapes
:`
V5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V5/strided_sliceStridedSliceV5/Shape:output:0V5/strided_slice/stack:output:0!V5/strided_slice/stack_1:output:0!V5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V5/Reshape/shapePackV5/strided_slice:output:0V5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V5/ReshapeReshapefeatures_v5V5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V6/ShapeShapefeatures_v6*
T0*
_output_shapes
:`
V6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V6/strided_sliceStridedSliceV6/Shape:output:0V6/strided_slice/stack:output:0!V6/strided_slice/stack_1:output:0!V6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V6/Reshape/shapePackV6/strided_slice:output:0V6/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V6/ReshapeReshapefeatures_v6V6/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V7/ShapeShapefeatures_v7*
T0*
_output_shapes
:`
V7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V7/strided_sliceStridedSliceV7/Shape:output:0V7/strided_slice/stack:output:0!V7/strided_slice/stack_1:output:0!V7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V7/Reshape/shapePackV7/strided_slice:output:0V7/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V7/ReshapeReshapefeatures_v7V7/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V8/ShapeShapefeatures_v8*
T0*
_output_shapes
:`
V8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V8/strided_sliceStridedSliceV8/Shape:output:0V8/strided_slice/stack:output:0!V8/strided_slice/stack_1:output:0!V8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V8/Reshape/shapePackV8/strided_slice:output:0V8/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V8/ReshapeReshapefeatures_v8V8/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????C
V9/ShapeShapefeatures_v9*
T0*
_output_shapes
:`
V9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: b
V9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
V9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
V9/strided_sliceStridedSliceV9/Shape:output:0V9/strided_slice/stack:output:0!V9/strided_slice/stack_1:output:0!V9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
V9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
V9/Reshape/shapePackV9/strided_slice:output:0V9/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:o

V9/ReshapeReshapefeatures_v9V9/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2Amount/Reshape:output:0Time/Reshape:output:0V1/Reshape:output:0V10/Reshape:output:0V11/Reshape:output:0V12/Reshape:output:0V13/Reshape:output:0V14/Reshape:output:0V15/Reshape:output:0V16/Reshape:output:0V17/Reshape:output:0V18/Reshape:output:0V19/Reshape:output:0V2/Reshape:output:0V20/Reshape:output:0V21/Reshape:output:0V22/Reshape:output:0V23/Reshape:output:0V24/Reshape:output:0V25/Reshape:output:0V26/Reshape:output:0V27/Reshape:output:0V28/Reshape:output:0V3/Reshape:output:0V4/Reshape:output:0V5/Reshape:output:0V6/Reshape:output:0V7/Reshape:output:0V8/Reshape:output:0V9/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:X T
'
_output_shapes
:?????????
)
_user_specified_namefeatures/Amount:VR
'
_output_shapes
:?????????
'
_user_specified_namefeatures/Time:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V1:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V10:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V11:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V12:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V13:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V14:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V15:U	Q
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V16:U
Q
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V17:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V18:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V19:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V2:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V20:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V21:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V22:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V23:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V24:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V25:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V26:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V27:UQ
'
_output_shapes
:?????????
&
_user_specified_namefeatures/V28:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V3:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V4:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V5:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V6:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V7:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V8:TP
'
_output_shapes
:?????????
%
_user_specified_namefeatures/V9
?Y
?
"__inference__traced_restore_142236
file_prefix8
*assignvariableop_batch_normalization_gamma:9
+assignvariableop_1_batch_normalization_beta:@
2assignvariableop_2_batch_normalization_moving_mean:D
6assignvariableop_3_batch_normalization_moving_variance:1
assignvariableop_4_dense_kernel:+
assignvariableop_5_dense_bias:%
assignvariableop_6_sgd_iter:	 &
assignvariableop_7_sgd_decay: .
$assignvariableop_8_sgd_learning_rate: )
assignvariableop_9_sgd_momentum: #
assignvariableop_10_total: #
assignvariableop_11_count: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: 1
"assignvariableop_14_true_positives:	?1
"assignvariableop_15_true_negatives:	?2
#assignvariableop_16_false_positives:	?2
#assignvariableop_17_false_negatives:	?H
:assignvariableop_18_sgd_batch_normalization_gamma_momentum:G
9assignvariableop_19_sgd_batch_normalization_beta_momentum:?
-assignvariableop_20_sgd_dense_kernel_momentum:9
+assignvariableop_21_sgd_dense_bias_momentum:
identity_23??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?
B?
B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp*assignvariableop_batch_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp+assignvariableop_1_batch_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp2assignvariableop_2_batch_normalization_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp6assignvariableop_3_batch_normalization_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_sgd_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_sgd_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp$assignvariableop_8_sgd_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_momentumIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_true_positivesIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp"assignvariableop_15_true_negativesIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_false_positivesIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp#assignvariableop_17_false_negativesIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp:assignvariableop_18_sgd_batch_normalization_gamma_momentumIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp9assignvariableop_19_sgd_batch_normalization_beta_momentumIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp-assignvariableop_20_sgd_dense_kernel_momentumIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_sgd_dense_bias_momentumIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_23Identity_23:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
9
Amount/
serving_default_Amount:0?????????
5
Time-
serving_default_Time:0?????????
1
V1+
serving_default_V1:0?????????
3
V10,
serving_default_V10:0?????????
3
V11,
serving_default_V11:0?????????
3
V12,
serving_default_V12:0?????????
3
V13,
serving_default_V13:0?????????
3
V14,
serving_default_V14:0?????????
3
V15,
serving_default_V15:0?????????
3
V16,
serving_default_V16:0?????????
3
V17,
serving_default_V17:0?????????
3
V18,
serving_default_V18:0?????????
3
V19,
serving_default_V19:0?????????
1
V2+
serving_default_V2:0?????????
3
V20,
serving_default_V20:0?????????
3
V21,
serving_default_V21:0?????????
3
V22,
serving_default_V22:0?????????
3
V23,
serving_default_V23:0?????????
3
V24,
serving_default_V24:0?????????
3
V25,
serving_default_V25:0?????????
3
V26,
serving_default_V26:0?????????
3
V27,
serving_default_V27:0?????????
3
V28,
serving_default_V28:0?????????
1
V3+
serving_default_V3:0?????????
1
V4+
serving_default_V4:0?????????
1
V5+
serving_default_V5:0?????????
1
V6+
serving_default_V6:0?????????
1
V7+
serving_default_V7:0?????????
1
V8+
serving_default_V8:0?????????
1
V9+
serving_default_V9:0?????????9
dense0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer_with_weights-0
 layer-31
!layer_with_weights-1
!layer-32
"	optimizer
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'
signatures
k__call__
*l&call_and_return_all_conditional_losses
m_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
?
(_feature_columns
)
_resources
*	variables
+trainable_variables
,regularization_losses
-	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
?
.axis
	/gamma
0beta
1moving_mean
2moving_variance
3	variables
4trainable_variables
5regularization_losses
6	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer
?

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
r__call__
*s&call_and_return_all_conditional_losses"
_tf_keras_layer
?
=iter
	>decay
?learning_rate
@momentum/momentumg0momentumh7momentumi8momentumj"
	optimizer
J
/0
01
12
23
74
85"
trackable_list_wrapper
<
/0
01
72
83"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
#	variables
$trainable_variables
%regularization_losses
k__call__
m_default_save_signature
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
,
tserving_default"
signature_map
 "
trackable_list_wrapper
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
*	variables
+trainable_variables
,regularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
<
/0
01
12
23"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
3	variables
4trainable_variables
5regularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
:2dense/kernel
:2
dense/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
9	variables
:trainable_variables
;regularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
.
10
21"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32"
trackable_list_wrapper
5
U0
V1
W2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
N
	Xtotal
	Ycount
Z	variables
[	keras_api"
_tf_keras_metric
^
	\total
	]count
^
_fn_kwargs
_	variables
`	keras_api"
_tf_keras_metric
?
atrue_positives
btrue_negatives
cfalse_positives
dfalse_negatives
e	variables
f	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
X0
Y1"
trackable_list_wrapper
-
Z	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
\0
]1"
trackable_list_wrapper
-
_	variables"
_generic_user_object
:? (2true_positives
:? (2true_negatives
 :? (2false_positives
 :? (2false_negatives
<
a0
b1
c2
d3"
trackable_list_wrapper
-
e	variables"
_generic_user_object
2:02&SGD/batch_normalization/gamma/momentum
1:/2%SGD/batch_normalization/beta/momentum
):'2SGD/dense/kernel/momentum
#:!2SGD/dense/bias/momentum
?2?
&__inference_model_layer_call_fn_139934
&__inference_model_layer_call_fn_140668
&__inference_model_layer_call_fn_140714
&__inference_model_layer_call_fn_140474?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_model_layer_call_and_return_conditional_losses_141012
A__inference_model_layer_call_and_return_conditional_losses_141324
A__inference_model_layer_call_and_return_conditional_losses_140522
A__inference_model_layer_call_and_return_conditional_losses_140570?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_139468AmountTimeV1V10V11V12V13V14V15V16V17V18V19V2V20V21V22V23V24V25V26V27V28V3V4V5V6V7V8V9"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_dense_features_layer_call_fn_141358
/__inference_dense_features_layer_call_fn_141392?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_dense_features_layer_call_and_return_conditional_losses_141667
J__inference_dense_features_layer_call_and_return_conditional_losses_141942?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_batch_normalization_layer_call_fn_141955
4__inference_batch_normalization_layer_call_fn_141968?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_141988
O__inference_batch_normalization_layer_call_and_return_conditional_losses_142022?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_dense_layer_call_fn_142031?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_layer_call_and_return_conditional_losses_142042?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_140622AmountTimeV1V10V11V12V13V14V15V16V17V18V19V2V20V21V22V23V24V25V26V27V28V3V4V5V6V7V8V9"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?	
!__inference__wrapped_model_139468?	2/1078?	??
???
???
*
Amount ?
Amount?????????
&
Time?
Time?????????
"
V1?
V1?????????
$
V10?
V10?????????
$
V11?
V11?????????
$
V12?
V12?????????
$
V13?
V13?????????
$
V14?
V14?????????
$
V15?
V15?????????
$
V16?
V16?????????
$
V17?
V17?????????
$
V18?
V18?????????
$
V19?
V19?????????
"
V2?
V2?????????
$
V20?
V20?????????
$
V21?
V21?????????
$
V22?
V22?????????
$
V23?
V23?????????
$
V24?
V24?????????
$
V25?
V25?????????
$
V26?
V26?????????
$
V27?
V27?????????
$
V28?
V28?????????
"
V3?
V3?????????
"
V4?
V4?????????
"
V5?
V5?????????
"
V6?
V6?????????
"
V7?
V7?????????
"
V8?
V8?????????
"
V9?
V9?????????
? "-?*
(
dense?
dense??????????
O__inference_batch_normalization_layer_call_and_return_conditional_losses_141988b2/103?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_142022b12/03?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
4__inference_batch_normalization_layer_call_fn_141955U2/103?0
)?&
 ?
inputs?????????
p 
? "???????????
4__inference_batch_normalization_layer_call_fn_141968U12/03?0
)?&
 ?
inputs?????????
p
? "???????????
J__inference_dense_features_layer_call_and_return_conditional_losses_141667????
???
?
??

3
Amount)?&
features/Amount?????????
/
Time'?$
features/Time?????????
+
V1%?"
features/V1?????????
-
V10&?#
features/V10?????????
-
V11&?#
features/V11?????????
-
V12&?#
features/V12?????????
-
V13&?#
features/V13?????????
-
V14&?#
features/V14?????????
-
V15&?#
features/V15?????????
-
V16&?#
features/V16?????????
-
V17&?#
features/V17?????????
-
V18&?#
features/V18?????????
-
V19&?#
features/V19?????????
+
V2%?"
features/V2?????????
-
V20&?#
features/V20?????????
-
V21&?#
features/V21?????????
-
V22&?#
features/V22?????????
-
V23&?#
features/V23?????????
-
V24&?#
features/V24?????????
-
V25&?#
features/V25?????????
-
V26&?#
features/V26?????????
-
V27&?#
features/V27?????????
-
V28&?#
features/V28?????????
+
V3%?"
features/V3?????????
+
V4%?"
features/V4?????????
+
V5%?"
features/V5?????????
+
V6%?"
features/V6?????????
+
V7%?"
features/V7?????????
+
V8%?"
features/V8?????????
+
V9%?"
features/V9?????????

 
p 
? "%?"
?
0?????????
? ?
J__inference_dense_features_layer_call_and_return_conditional_losses_141942????
???
?
??

3
Amount)?&
features/Amount?????????
/
Time'?$
features/Time?????????
+
V1%?"
features/V1?????????
-
V10&?#
features/V10?????????
-
V11&?#
features/V11?????????
-
V12&?#
features/V12?????????
-
V13&?#
features/V13?????????
-
V14&?#
features/V14?????????
-
V15&?#
features/V15?????????
-
V16&?#
features/V16?????????
-
V17&?#
features/V17?????????
-
V18&?#
features/V18?????????
-
V19&?#
features/V19?????????
+
V2%?"
features/V2?????????
-
V20&?#
features/V20?????????
-
V21&?#
features/V21?????????
-
V22&?#
features/V22?????????
-
V23&?#
features/V23?????????
-
V24&?#
features/V24?????????
-
V25&?#
features/V25?????????
-
V26&?#
features/V26?????????
-
V27&?#
features/V27?????????
-
V28&?#
features/V28?????????
+
V3%?"
features/V3?????????
+
V4%?"
features/V4?????????
+
V5%?"
features/V5?????????
+
V6%?"
features/V6?????????
+
V7%?"
features/V7?????????
+
V8%?"
features/V8?????????
+
V9%?"
features/V9?????????

 
p
? "%?"
?
0?????????
? ?
/__inference_dense_features_layer_call_fn_141358????
???
?
??

3
Amount)?&
features/Amount?????????
/
Time'?$
features/Time?????????
+
V1%?"
features/V1?????????
-
V10&?#
features/V10?????????
-
V11&?#
features/V11?????????
-
V12&?#
features/V12?????????
-
V13&?#
features/V13?????????
-
V14&?#
features/V14?????????
-
V15&?#
features/V15?????????
-
V16&?#
features/V16?????????
-
V17&?#
features/V17?????????
-
V18&?#
features/V18?????????
-
V19&?#
features/V19?????????
+
V2%?"
features/V2?????????
-
V20&?#
features/V20?????????
-
V21&?#
features/V21?????????
-
V22&?#
features/V22?????????
-
V23&?#
features/V23?????????
-
V24&?#
features/V24?????????
-
V25&?#
features/V25?????????
-
V26&?#
features/V26?????????
-
V27&?#
features/V27?????????
-
V28&?#
features/V28?????????
+
V3%?"
features/V3?????????
+
V4%?"
features/V4?????????
+
V5%?"
features/V5?????????
+
V6%?"
features/V6?????????
+
V7%?"
features/V7?????????
+
V8%?"
features/V8?????????
+
V9%?"
features/V9?????????

 
p 
? "???????????
/__inference_dense_features_layer_call_fn_141392????
???
?
??

3
Amount)?&
features/Amount?????????
/
Time'?$
features/Time?????????
+
V1%?"
features/V1?????????
-
V10&?#
features/V10?????????
-
V11&?#
features/V11?????????
-
V12&?#
features/V12?????????
-
V13&?#
features/V13?????????
-
V14&?#
features/V14?????????
-
V15&?#
features/V15?????????
-
V16&?#
features/V16?????????
-
V17&?#
features/V17?????????
-
V18&?#
features/V18?????????
-
V19&?#
features/V19?????????
+
V2%?"
features/V2?????????
-
V20&?#
features/V20?????????
-
V21&?#
features/V21?????????
-
V22&?#
features/V22?????????
-
V23&?#
features/V23?????????
-
V24&?#
features/V24?????????
-
V25&?#
features/V25?????????
-
V26&?#
features/V26?????????
-
V27&?#
features/V27?????????
-
V28&?#
features/V28?????????
+
V3%?"
features/V3?????????
+
V4%?"
features/V4?????????
+
V5%?"
features/V5?????????
+
V6%?"
features/V6?????????
+
V7%?"
features/V7?????????
+
V8%?"
features/V8?????????
+
V9%?"
features/V9?????????

 
p
? "???????????
A__inference_dense_layer_call_and_return_conditional_losses_142042\78/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
&__inference_dense_layer_call_fn_142031O78/?,
%?"
 ?
inputs?????????
? "???????????

A__inference_model_layer_call_and_return_conditional_losses_140522?	2/1078?	??	
???
???
*
Amount ?
Amount?????????
&
Time?
Time?????????
"
V1?
V1?????????
$
V10?
V10?????????
$
V11?
V11?????????
$
V12?
V12?????????
$
V13?
V13?????????
$
V14?
V14?????????
$
V15?
V15?????????
$
V16?
V16?????????
$
V17?
V17?????????
$
V18?
V18?????????
$
V19?
V19?????????
"
V2?
V2?????????
$
V20?
V20?????????
$
V21?
V21?????????
$
V22?
V22?????????
$
V23?
V23?????????
$
V24?
V24?????????
$
V25?
V25?????????
$
V26?
V26?????????
$
V27?
V27?????????
$
V28?
V28?????????
"
V3?
V3?????????
"
V4?
V4?????????
"
V5?
V5?????????
"
V6?
V6?????????
"
V7?
V7?????????
"
V8?
V8?????????
"
V9?
V9?????????
p 

 
? "%?"
?
0?????????
? ?

A__inference_model_layer_call_and_return_conditional_losses_140570?	12/078?	??	
???
???
*
Amount ?
Amount?????????
&
Time?
Time?????????
"
V1?
V1?????????
$
V10?
V10?????????
$
V11?
V11?????????
$
V12?
V12?????????
$
V13?
V13?????????
$
V14?
V14?????????
$
V15?
V15?????????
$
V16?
V16?????????
$
V17?
V17?????????
$
V18?
V18?????????
$
V19?
V19?????????
"
V2?
V2?????????
$
V20?
V20?????????
$
V21?
V21?????????
$
V22?
V22?????????
$
V23?
V23?????????
$
V24?
V24?????????
$
V25?
V25?????????
$
V26?
V26?????????
$
V27?
V27?????????
$
V28?
V28?????????
"
V3?
V3?????????
"
V4?
V4?????????
"
V5?
V5?????????
"
V6?
V6?????????
"
V7?
V7?????????
"
V8?
V8?????????
"
V9?
V9?????????
p

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_141012?2/1078?
??

?
??

?
??

1
Amount'?$
inputs/Amount?????????
-
Time%?"
inputs/Time?????????
)
V1#? 
	inputs/V1?????????
+
V10$?!

inputs/V10?????????
+
V11$?!

inputs/V11?????????
+
V12$?!

inputs/V12?????????
+
V13$?!

inputs/V13?????????
+
V14$?!

inputs/V14?????????
+
V15$?!

inputs/V15?????????
+
V16$?!

inputs/V16?????????
+
V17$?!

inputs/V17?????????
+
V18$?!

inputs/V18?????????
+
V19$?!

inputs/V19?????????
)
V2#? 
	inputs/V2?????????
+
V20$?!

inputs/V20?????????
+
V21$?!

inputs/V21?????????
+
V22$?!

inputs/V22?????????
+
V23$?!

inputs/V23?????????
+
V24$?!

inputs/V24?????????
+
V25$?!

inputs/V25?????????
+
V26$?!

inputs/V26?????????
+
V27$?!

inputs/V27?????????
+
V28$?!

inputs/V28?????????
)
V3#? 
	inputs/V3?????????
)
V4#? 
	inputs/V4?????????
)
V5#? 
	inputs/V5?????????
)
V6#? 
	inputs/V6?????????
)
V7#? 
	inputs/V7?????????
)
V8#? 
	inputs/V8?????????
)
V9#? 
	inputs/V9?????????
p 

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_141324?12/078?
??

?
??

?
??

1
Amount'?$
inputs/Amount?????????
-
Time%?"
inputs/Time?????????
)
V1#? 
	inputs/V1?????????
+
V10$?!

inputs/V10?????????
+
V11$?!

inputs/V11?????????
+
V12$?!

inputs/V12?????????
+
V13$?!

inputs/V13?????????
+
V14$?!

inputs/V14?????????
+
V15$?!

inputs/V15?????????
+
V16$?!

inputs/V16?????????
+
V17$?!

inputs/V17?????????
+
V18$?!

inputs/V18?????????
+
V19$?!

inputs/V19?????????
)
V2#? 
	inputs/V2?????????
+
V20$?!

inputs/V20?????????
+
V21$?!

inputs/V21?????????
+
V22$?!

inputs/V22?????????
+
V23$?!

inputs/V23?????????
+
V24$?!

inputs/V24?????????
+
V25$?!

inputs/V25?????????
+
V26$?!

inputs/V26?????????
+
V27$?!

inputs/V27?????????
+
V28$?!

inputs/V28?????????
)
V3#? 
	inputs/V3?????????
)
V4#? 
	inputs/V4?????????
)
V5#? 
	inputs/V5?????????
)
V6#? 
	inputs/V6?????????
)
V7#? 
	inputs/V7?????????
)
V8#? 
	inputs/V8?????????
)
V9#? 
	inputs/V9?????????
p

 
? "%?"
?
0?????????
? ?	
&__inference_model_layer_call_fn_139934?	2/1078?	??	
???
???
*
Amount ?
Amount?????????
&
Time?
Time?????????
"
V1?
V1?????????
$
V10?
V10?????????
$
V11?
V11?????????
$
V12?
V12?????????
$
V13?
V13?????????
$
V14?
V14?????????
$
V15?
V15?????????
$
V16?
V16?????????
$
V17?
V17?????????
$
V18?
V18?????????
$
V19?
V19?????????
"
V2?
V2?????????
$
V20?
V20?????????
$
V21?
V21?????????
$
V22?
V22?????????
$
V23?
V23?????????
$
V24?
V24?????????
$
V25?
V25?????????
$
V26?
V26?????????
$
V27?
V27?????????
$
V28?
V28?????????
"
V3?
V3?????????
"
V4?
V4?????????
"
V5?
V5?????????
"
V6?
V6?????????
"
V7?
V7?????????
"
V8?
V8?????????
"
V9?
V9?????????
p 

 
? "???????????	
&__inference_model_layer_call_fn_140474?	12/078?	??	
???
???
*
Amount ?
Amount?????????
&
Time?
Time?????????
"
V1?
V1?????????
$
V10?
V10?????????
$
V11?
V11?????????
$
V12?
V12?????????
$
V13?
V13?????????
$
V14?
V14?????????
$
V15?
V15?????????
$
V16?
V16?????????
$
V17?
V17?????????
$
V18?
V18?????????
$
V19?
V19?????????
"
V2?
V2?????????
$
V20?
V20?????????
$
V21?
V21?????????
$
V22?
V22?????????
$
V23?
V23?????????
$
V24?
V24?????????
$
V25?
V25?????????
$
V26?
V26?????????
$
V27?
V27?????????
$
V28?
V28?????????
"
V3?
V3?????????
"
V4?
V4?????????
"
V5?
V5?????????
"
V6?
V6?????????
"
V7?
V7?????????
"
V8?
V8?????????
"
V9?
V9?????????
p

 
? "???????????
&__inference_model_layer_call_fn_140668?2/1078?
??

?
??

?
??

1
Amount'?$
inputs/Amount?????????
-
Time%?"
inputs/Time?????????
)
V1#? 
	inputs/V1?????????
+
V10$?!

inputs/V10?????????
+
V11$?!

inputs/V11?????????
+
V12$?!

inputs/V12?????????
+
V13$?!

inputs/V13?????????
+
V14$?!

inputs/V14?????????
+
V15$?!

inputs/V15?????????
+
V16$?!

inputs/V16?????????
+
V17$?!

inputs/V17?????????
+
V18$?!

inputs/V18?????????
+
V19$?!

inputs/V19?????????
)
V2#? 
	inputs/V2?????????
+
V20$?!

inputs/V20?????????
+
V21$?!

inputs/V21?????????
+
V22$?!

inputs/V22?????????
+
V23$?!

inputs/V23?????????
+
V24$?!

inputs/V24?????????
+
V25$?!

inputs/V25?????????
+
V26$?!

inputs/V26?????????
+
V27$?!

inputs/V27?????????
+
V28$?!

inputs/V28?????????
)
V3#? 
	inputs/V3?????????
)
V4#? 
	inputs/V4?????????
)
V5#? 
	inputs/V5?????????
)
V6#? 
	inputs/V6?????????
)
V7#? 
	inputs/V7?????????
)
V8#? 
	inputs/V8?????????
)
V9#? 
	inputs/V9?????????
p 

 
? "???????????
&__inference_model_layer_call_fn_140714?12/078?
??

?
??

?
??

1
Amount'?$
inputs/Amount?????????
-
Time%?"
inputs/Time?????????
)
V1#? 
	inputs/V1?????????
+
V10$?!

inputs/V10?????????
+
V11$?!

inputs/V11?????????
+
V12$?!

inputs/V12?????????
+
V13$?!

inputs/V13?????????
+
V14$?!

inputs/V14?????????
+
V15$?!

inputs/V15?????????
+
V16$?!

inputs/V16?????????
+
V17$?!

inputs/V17?????????
+
V18$?!

inputs/V18?????????
+
V19$?!

inputs/V19?????????
)
V2#? 
	inputs/V2?????????
+
V20$?!

inputs/V20?????????
+
V21$?!

inputs/V21?????????
+
V22$?!

inputs/V22?????????
+
V23$?!

inputs/V23?????????
+
V24$?!

inputs/V24?????????
+
V25$?!

inputs/V25?????????
+
V26$?!

inputs/V26?????????
+
V27$?!

inputs/V27?????????
+
V28$?!

inputs/V28?????????
)
V3#? 
	inputs/V3?????????
)
V4#? 
	inputs/V4?????????
)
V5#? 
	inputs/V5?????????
)
V6#? 
	inputs/V6?????????
)
V7#? 
	inputs/V7?????????
)
V8#? 
	inputs/V8?????????
)
V9#? 
	inputs/V9?????????
p

 
? "???????????	
$__inference_signature_wrapper_140622?	2/1078???
? 
???
*
Amount ?
Amount?????????
&
Time?
Time?????????
"
V1?
V1?????????
$
V10?
V10?????????
$
V11?
V11?????????
$
V12?
V12?????????
$
V13?
V13?????????
$
V14?
V14?????????
$
V15?
V15?????????
$
V16?
V16?????????
$
V17?
V17?????????
$
V18?
V18?????????
$
V19?
V19?????????
"
V2?
V2?????????
$
V20?
V20?????????
$
V21?
V21?????????
$
V22?
V22?????????
$
V23?
V23?????????
$
V24?
V24?????????
$
V25?
V25?????????
$
V26?
V26?????????
$
V27?
V27?????????
$
V28?
V28?????????
"
V3?
V3?????????
"
V4?
V4?????????
"
V5?
V5?????????
"
V6?
V6?????????
"
V7?
V7?????????
"
V8?
V8?????????
"
V9?
V9?????????"-?*
(
dense?
dense?????????