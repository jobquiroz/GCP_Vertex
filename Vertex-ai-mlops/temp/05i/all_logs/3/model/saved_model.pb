 
Í£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.42v2.3.4-0-gea90cf44f738êÃ

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance

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
shape:È*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:È*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:È*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:È*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:È*
dtype0
¤
&SGD/batch_normalization/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&SGD/batch_normalization/gamma/momentum

:SGD/batch_normalization/gamma/momentum/Read/ReadVariableOpReadVariableOp&SGD/batch_normalization/gamma/momentum*
_output_shapes
:*
dtype0
¢
%SGD/batch_normalization/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%SGD/batch_normalization/beta/momentum

9SGD/batch_normalization/beta/momentum/Read/ReadVariableOpReadVariableOp%SGD/batch_normalization/beta/momentum*
_output_shapes
:*
dtype0

SGD/dense/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameSGD/dense/kernel/momentum

-SGD/dense/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense/kernel/momentum*
_output_shapes

:*
dtype0

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
Ü'
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*'
value'B' B'
é
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
$regularization_losses
%trainable_variables
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
+regularization_losses
,trainable_variables
-	keras_api

.axis
	/gamma
0beta
1moving_mean
2moving_variance
3	variables
4regularization_losses
5trainable_variables
6	keras_api
h

7kernel
8bias
9	variables
:regularization_losses
;trainable_variables
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
 

/0
01
72
83
­

Alayers
Bnon_trainable_variables
Clayer_regularization_losses
#	variables
$regularization_losses
Dlayer_metrics
Emetrics
%trainable_variables
 
 
 
 
 
 
­

Flayers
Gmetrics
Hnon_trainable_variables
Ilayer_regularization_losses
*	variables
+regularization_losses
Jlayer_metrics
,trainable_variables
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
 

/0
01
­

Klayers
Lmetrics
Mnon_trainable_variables
Nlayer_regularization_losses
3	variables
4regularization_losses
Olayer_metrics
5trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

70
81
 

70
81
­

Players
Qmetrics
Rnon_trainable_variables
Slayer_regularization_losses
9	variables
:regularization_losses
Tlayer_metrics
;trainable_variables
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
þ
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

10
21
 
 
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

VARIABLE_VALUE&SGD/batch_normalization/gamma/momentumXlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE%SGD/batch_normalization/beta/momentumWlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
y
serving_default_AmountPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
w
serving_default_TimePlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
u
serving_default_V1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_V10Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_V11Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_V12Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_V13Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_V14Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_V15Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_V16Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_V17Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_V18Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_V19Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
u
serving_default_V2Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_V20Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_V21Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_V22Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_V23Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_V24Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_V25Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_V26Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_V27Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_V28Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
u
serving_default_V3Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
u
serving_default_V4Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
u
serving_default_V5Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
u
serving_default_V6Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
u
serving_default_V7Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
u
serving_default_V8Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
u
serving_default_V9Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
Ç
StatefulPartitionedCallStatefulPartitionedCallserving_default_Amountserving_default_Timeserving_default_V1serving_default_V10serving_default_V11serving_default_V12serving_default_V13serving_default_V14serving_default_V15serving_default_V16serving_default_V17serving_default_V18serving_default_V19serving_default_V2serving_default_V20serving_default_V21serving_default_V22serving_default_V23serving_default_V24serving_default_V25serving_default_V26serving_default_V27serving_default_V28serving_default_V3serving_default_V4serving_default_V5serving_default_V6serving_default_V7serving_default_V8serving_default_V9#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense/kernel
dense/bias*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

 !"#*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_82696
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
	
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
GPU 2J 8 *'
f"R 
__inference__traced_save_84238
Û
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
GPU 2J 8 **
f%R#
!__inference__traced_restore_84314þ½
ô%

,__inference_functional_1_layer_call_fn_83400
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
	inputs_v9
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall£
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
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

 !"#*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_826272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*ç
_input_shapesÕ
Ò:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameinputs/Amount:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinputs/Time:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V1:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V10:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V11:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V12:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V13:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V14:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V15:S	O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V16:S
O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V17:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V18:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V19:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V2:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V20:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V21:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V22:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V23:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V24:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V25:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V26:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V27:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V28:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V3:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V4:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V5:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V6:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V7:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V8:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V9
©
Ë
I__inference_dense_features_layer_call_and_return_conditional_losses_81996
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
identityT
Amount/ShapeShapefeatures*
T0*
_output_shapes
:2
Amount/Shape
Amount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Amount/strided_slice/stack
Amount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Amount/strided_slice/stack_1
Amount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Amount/strided_slice/stack_2
Amount/strided_sliceStridedSliceAmount/Shape:output:0#Amount/strided_slice/stack:output:0%Amount/strided_slice/stack_1:output:0%Amount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Amount/strided_slicer
Amount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Amount/Reshape/shape/1¢
Amount/Reshape/shapePackAmount/strided_slice:output:0Amount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Amount/Reshape/shape
Amount/ReshapeReshapefeaturesAmount/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Amount/ReshapeR

Time/ShapeShape
features_1*
T0*
_output_shapes
:2

Time/Shape~
Time/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Time/strided_slice/stack
Time/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Time/strided_slice/stack_1
Time/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Time/strided_slice/stack_2
Time/strided_sliceStridedSliceTime/Shape:output:0!Time/strided_slice/stack:output:0#Time/strided_slice/stack_1:output:0#Time/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Time/strided_slicen
Time/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Time/Reshape/shape/1
Time/Reshape/shapePackTime/strided_slice:output:0Time/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Time/Reshape/shape
Time/ReshapeReshape
features_1Time/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Time/ReshapeN
V1/ShapeShape
features_2*
T0*
_output_shapes
:2

V1/Shapez
V1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V1/strided_slice/stack~
V1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V1/strided_slice/stack_1~
V1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V1/strided_slice/stack_2ô
V1/strided_sliceStridedSliceV1/Shape:output:0V1/strided_slice/stack:output:0!V1/strided_slice/stack_1:output:0!V1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V1/strided_slicej
V1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V1/Reshape/shape/1
V1/Reshape/shapePackV1/strided_slice:output:0V1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V1/Reshape/shape|

V1/ReshapeReshape
features_2V1/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V1/ReshapeP
	V10/ShapeShape
features_3*
T0*
_output_shapes
:2
	V10/Shape|
V10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V10/strided_slice/stack
V10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V10/strided_slice/stack_1
V10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V10/strided_slice/stack_2ú
V10/strided_sliceStridedSliceV10/Shape:output:0 V10/strided_slice/stack:output:0"V10/strided_slice/stack_1:output:0"V10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V10/strided_slicel
V10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V10/Reshape/shape/1
V10/Reshape/shapePackV10/strided_slice:output:0V10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V10/Reshape/shape
V10/ReshapeReshape
features_3V10/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V10/ReshapeP
	V11/ShapeShape
features_4*
T0*
_output_shapes
:2
	V11/Shape|
V11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V11/strided_slice/stack
V11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V11/strided_slice/stack_1
V11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V11/strided_slice/stack_2ú
V11/strided_sliceStridedSliceV11/Shape:output:0 V11/strided_slice/stack:output:0"V11/strided_slice/stack_1:output:0"V11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V11/strided_slicel
V11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V11/Reshape/shape/1
V11/Reshape/shapePackV11/strided_slice:output:0V11/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V11/Reshape/shape
V11/ReshapeReshape
features_4V11/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V11/ReshapeP
	V12/ShapeShape
features_5*
T0*
_output_shapes
:2
	V12/Shape|
V12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V12/strided_slice/stack
V12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V12/strided_slice/stack_1
V12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V12/strided_slice/stack_2ú
V12/strided_sliceStridedSliceV12/Shape:output:0 V12/strided_slice/stack:output:0"V12/strided_slice/stack_1:output:0"V12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V12/strided_slicel
V12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V12/Reshape/shape/1
V12/Reshape/shapePackV12/strided_slice:output:0V12/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V12/Reshape/shape
V12/ReshapeReshape
features_5V12/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V12/ReshapeP
	V13/ShapeShape
features_6*
T0*
_output_shapes
:2
	V13/Shape|
V13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V13/strided_slice/stack
V13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V13/strided_slice/stack_1
V13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V13/strided_slice/stack_2ú
V13/strided_sliceStridedSliceV13/Shape:output:0 V13/strided_slice/stack:output:0"V13/strided_slice/stack_1:output:0"V13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V13/strided_slicel
V13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V13/Reshape/shape/1
V13/Reshape/shapePackV13/strided_slice:output:0V13/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V13/Reshape/shape
V13/ReshapeReshape
features_6V13/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V13/ReshapeP
	V14/ShapeShape
features_7*
T0*
_output_shapes
:2
	V14/Shape|
V14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V14/strided_slice/stack
V14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V14/strided_slice/stack_1
V14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V14/strided_slice/stack_2ú
V14/strided_sliceStridedSliceV14/Shape:output:0 V14/strided_slice/stack:output:0"V14/strided_slice/stack_1:output:0"V14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V14/strided_slicel
V14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V14/Reshape/shape/1
V14/Reshape/shapePackV14/strided_slice:output:0V14/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V14/Reshape/shape
V14/ReshapeReshape
features_7V14/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V14/ReshapeP
	V15/ShapeShape
features_8*
T0*
_output_shapes
:2
	V15/Shape|
V15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V15/strided_slice/stack
V15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V15/strided_slice/stack_1
V15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V15/strided_slice/stack_2ú
V15/strided_sliceStridedSliceV15/Shape:output:0 V15/strided_slice/stack:output:0"V15/strided_slice/stack_1:output:0"V15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V15/strided_slicel
V15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V15/Reshape/shape/1
V15/Reshape/shapePackV15/strided_slice:output:0V15/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V15/Reshape/shape
V15/ReshapeReshape
features_8V15/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V15/ReshapeP
	V16/ShapeShape
features_9*
T0*
_output_shapes
:2
	V16/Shape|
V16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V16/strided_slice/stack
V16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V16/strided_slice/stack_1
V16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V16/strided_slice/stack_2ú
V16/strided_sliceStridedSliceV16/Shape:output:0 V16/strided_slice/stack:output:0"V16/strided_slice/stack_1:output:0"V16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V16/strided_slicel
V16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V16/Reshape/shape/1
V16/Reshape/shapePackV16/strided_slice:output:0V16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V16/Reshape/shape
V16/ReshapeReshape
features_9V16/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V16/ReshapeQ
	V17/ShapeShapefeatures_10*
T0*
_output_shapes
:2
	V17/Shape|
V17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V17/strided_slice/stack
V17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V17/strided_slice/stack_1
V17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V17/strided_slice/stack_2ú
V17/strided_sliceStridedSliceV17/Shape:output:0 V17/strided_slice/stack:output:0"V17/strided_slice/stack_1:output:0"V17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V17/strided_slicel
V17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V17/Reshape/shape/1
V17/Reshape/shapePackV17/strided_slice:output:0V17/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V17/Reshape/shape
V17/ReshapeReshapefeatures_10V17/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V17/ReshapeQ
	V18/ShapeShapefeatures_11*
T0*
_output_shapes
:2
	V18/Shape|
V18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V18/strided_slice/stack
V18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V18/strided_slice/stack_1
V18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V18/strided_slice/stack_2ú
V18/strided_sliceStridedSliceV18/Shape:output:0 V18/strided_slice/stack:output:0"V18/strided_slice/stack_1:output:0"V18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V18/strided_slicel
V18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V18/Reshape/shape/1
V18/Reshape/shapePackV18/strided_slice:output:0V18/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V18/Reshape/shape
V18/ReshapeReshapefeatures_11V18/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V18/ReshapeQ
	V19/ShapeShapefeatures_12*
T0*
_output_shapes
:2
	V19/Shape|
V19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V19/strided_slice/stack
V19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V19/strided_slice/stack_1
V19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V19/strided_slice/stack_2ú
V19/strided_sliceStridedSliceV19/Shape:output:0 V19/strided_slice/stack:output:0"V19/strided_slice/stack_1:output:0"V19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V19/strided_slicel
V19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V19/Reshape/shape/1
V19/Reshape/shapePackV19/strided_slice:output:0V19/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V19/Reshape/shape
V19/ReshapeReshapefeatures_12V19/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V19/ReshapeO
V2/ShapeShapefeatures_13*
T0*
_output_shapes
:2

V2/Shapez
V2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V2/strided_slice/stack~
V2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V2/strided_slice/stack_1~
V2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V2/strided_slice/stack_2ô
V2/strided_sliceStridedSliceV2/Shape:output:0V2/strided_slice/stack:output:0!V2/strided_slice/stack_1:output:0!V2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V2/strided_slicej
V2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V2/Reshape/shape/1
V2/Reshape/shapePackV2/strided_slice:output:0V2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V2/Reshape/shape}

V2/ReshapeReshapefeatures_13V2/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V2/ReshapeQ
	V20/ShapeShapefeatures_14*
T0*
_output_shapes
:2
	V20/Shape|
V20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V20/strided_slice/stack
V20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V20/strided_slice/stack_1
V20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V20/strided_slice/stack_2ú
V20/strided_sliceStridedSliceV20/Shape:output:0 V20/strided_slice/stack:output:0"V20/strided_slice/stack_1:output:0"V20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V20/strided_slicel
V20/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V20/Reshape/shape/1
V20/Reshape/shapePackV20/strided_slice:output:0V20/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V20/Reshape/shape
V20/ReshapeReshapefeatures_14V20/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V20/ReshapeQ
	V21/ShapeShapefeatures_15*
T0*
_output_shapes
:2
	V21/Shape|
V21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V21/strided_slice/stack
V21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V21/strided_slice/stack_1
V21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V21/strided_slice/stack_2ú
V21/strided_sliceStridedSliceV21/Shape:output:0 V21/strided_slice/stack:output:0"V21/strided_slice/stack_1:output:0"V21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V21/strided_slicel
V21/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V21/Reshape/shape/1
V21/Reshape/shapePackV21/strided_slice:output:0V21/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V21/Reshape/shape
V21/ReshapeReshapefeatures_15V21/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V21/ReshapeQ
	V22/ShapeShapefeatures_16*
T0*
_output_shapes
:2
	V22/Shape|
V22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V22/strided_slice/stack
V22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V22/strided_slice/stack_1
V22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V22/strided_slice/stack_2ú
V22/strided_sliceStridedSliceV22/Shape:output:0 V22/strided_slice/stack:output:0"V22/strided_slice/stack_1:output:0"V22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V22/strided_slicel
V22/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V22/Reshape/shape/1
V22/Reshape/shapePackV22/strided_slice:output:0V22/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V22/Reshape/shape
V22/ReshapeReshapefeatures_16V22/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V22/ReshapeQ
	V23/ShapeShapefeatures_17*
T0*
_output_shapes
:2
	V23/Shape|
V23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V23/strided_slice/stack
V23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V23/strided_slice/stack_1
V23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V23/strided_slice/stack_2ú
V23/strided_sliceStridedSliceV23/Shape:output:0 V23/strided_slice/stack:output:0"V23/strided_slice/stack_1:output:0"V23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V23/strided_slicel
V23/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V23/Reshape/shape/1
V23/Reshape/shapePackV23/strided_slice:output:0V23/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V23/Reshape/shape
V23/ReshapeReshapefeatures_17V23/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V23/ReshapeQ
	V24/ShapeShapefeatures_18*
T0*
_output_shapes
:2
	V24/Shape|
V24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V24/strided_slice/stack
V24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V24/strided_slice/stack_1
V24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V24/strided_slice/stack_2ú
V24/strided_sliceStridedSliceV24/Shape:output:0 V24/strided_slice/stack:output:0"V24/strided_slice/stack_1:output:0"V24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V24/strided_slicel
V24/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V24/Reshape/shape/1
V24/Reshape/shapePackV24/strided_slice:output:0V24/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V24/Reshape/shape
V24/ReshapeReshapefeatures_18V24/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V24/ReshapeQ
	V25/ShapeShapefeatures_19*
T0*
_output_shapes
:2
	V25/Shape|
V25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V25/strided_slice/stack
V25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V25/strided_slice/stack_1
V25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V25/strided_slice/stack_2ú
V25/strided_sliceStridedSliceV25/Shape:output:0 V25/strided_slice/stack:output:0"V25/strided_slice/stack_1:output:0"V25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V25/strided_slicel
V25/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V25/Reshape/shape/1
V25/Reshape/shapePackV25/strided_slice:output:0V25/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V25/Reshape/shape
V25/ReshapeReshapefeatures_19V25/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V25/ReshapeQ
	V26/ShapeShapefeatures_20*
T0*
_output_shapes
:2
	V26/Shape|
V26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V26/strided_slice/stack
V26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V26/strided_slice/stack_1
V26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V26/strided_slice/stack_2ú
V26/strided_sliceStridedSliceV26/Shape:output:0 V26/strided_slice/stack:output:0"V26/strided_slice/stack_1:output:0"V26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V26/strided_slicel
V26/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V26/Reshape/shape/1
V26/Reshape/shapePackV26/strided_slice:output:0V26/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V26/Reshape/shape
V26/ReshapeReshapefeatures_20V26/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V26/ReshapeQ
	V27/ShapeShapefeatures_21*
T0*
_output_shapes
:2
	V27/Shape|
V27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V27/strided_slice/stack
V27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V27/strided_slice/stack_1
V27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V27/strided_slice/stack_2ú
V27/strided_sliceStridedSliceV27/Shape:output:0 V27/strided_slice/stack:output:0"V27/strided_slice/stack_1:output:0"V27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V27/strided_slicel
V27/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V27/Reshape/shape/1
V27/Reshape/shapePackV27/strided_slice:output:0V27/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V27/Reshape/shape
V27/ReshapeReshapefeatures_21V27/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V27/ReshapeQ
	V28/ShapeShapefeatures_22*
T0*
_output_shapes
:2
	V28/Shape|
V28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V28/strided_slice/stack
V28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V28/strided_slice/stack_1
V28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V28/strided_slice/stack_2ú
V28/strided_sliceStridedSliceV28/Shape:output:0 V28/strided_slice/stack:output:0"V28/strided_slice/stack_1:output:0"V28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V28/strided_slicel
V28/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V28/Reshape/shape/1
V28/Reshape/shapePackV28/strided_slice:output:0V28/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V28/Reshape/shape
V28/ReshapeReshapefeatures_22V28/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V28/ReshapeO
V3/ShapeShapefeatures_23*
T0*
_output_shapes
:2

V3/Shapez
V3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V3/strided_slice/stack~
V3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V3/strided_slice/stack_1~
V3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V3/strided_slice/stack_2ô
V3/strided_sliceStridedSliceV3/Shape:output:0V3/strided_slice/stack:output:0!V3/strided_slice/stack_1:output:0!V3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V3/strided_slicej
V3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V3/Reshape/shape/1
V3/Reshape/shapePackV3/strided_slice:output:0V3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V3/Reshape/shape}

V3/ReshapeReshapefeatures_23V3/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V3/ReshapeO
V4/ShapeShapefeatures_24*
T0*
_output_shapes
:2

V4/Shapez
V4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V4/strided_slice/stack~
V4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V4/strided_slice/stack_1~
V4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V4/strided_slice/stack_2ô
V4/strided_sliceStridedSliceV4/Shape:output:0V4/strided_slice/stack:output:0!V4/strided_slice/stack_1:output:0!V4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V4/strided_slicej
V4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V4/Reshape/shape/1
V4/Reshape/shapePackV4/strided_slice:output:0V4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V4/Reshape/shape}

V4/ReshapeReshapefeatures_24V4/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V4/ReshapeO
V5/ShapeShapefeatures_25*
T0*
_output_shapes
:2

V5/Shapez
V5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V5/strided_slice/stack~
V5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V5/strided_slice/stack_1~
V5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V5/strided_slice/stack_2ô
V5/strided_sliceStridedSliceV5/Shape:output:0V5/strided_slice/stack:output:0!V5/strided_slice/stack_1:output:0!V5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V5/strided_slicej
V5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V5/Reshape/shape/1
V5/Reshape/shapePackV5/strided_slice:output:0V5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V5/Reshape/shape}

V5/ReshapeReshapefeatures_25V5/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V5/ReshapeO
V6/ShapeShapefeatures_26*
T0*
_output_shapes
:2

V6/Shapez
V6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V6/strided_slice/stack~
V6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V6/strided_slice/stack_1~
V6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V6/strided_slice/stack_2ô
V6/strided_sliceStridedSliceV6/Shape:output:0V6/strided_slice/stack:output:0!V6/strided_slice/stack_1:output:0!V6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V6/strided_slicej
V6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V6/Reshape/shape/1
V6/Reshape/shapePackV6/strided_slice:output:0V6/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V6/Reshape/shape}

V6/ReshapeReshapefeatures_26V6/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V6/ReshapeO
V7/ShapeShapefeatures_27*
T0*
_output_shapes
:2

V7/Shapez
V7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V7/strided_slice/stack~
V7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V7/strided_slice/stack_1~
V7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V7/strided_slice/stack_2ô
V7/strided_sliceStridedSliceV7/Shape:output:0V7/strided_slice/stack:output:0!V7/strided_slice/stack_1:output:0!V7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V7/strided_slicej
V7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V7/Reshape/shape/1
V7/Reshape/shapePackV7/strided_slice:output:0V7/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V7/Reshape/shape}

V7/ReshapeReshapefeatures_27V7/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V7/ReshapeO
V8/ShapeShapefeatures_28*
T0*
_output_shapes
:2

V8/Shapez
V8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V8/strided_slice/stack~
V8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V8/strided_slice/stack_1~
V8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V8/strided_slice/stack_2ô
V8/strided_sliceStridedSliceV8/Shape:output:0V8/strided_slice/stack:output:0!V8/strided_slice/stack_1:output:0!V8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V8/strided_slicej
V8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V8/Reshape/shape/1
V8/Reshape/shapePackV8/strided_slice:output:0V8/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V8/Reshape/shape}

V8/ReshapeReshapefeatures_28V8/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V8/ReshapeO
V9/ShapeShapefeatures_29*
T0*
_output_shapes
:2

V9/Shapez
V9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V9/strided_slice/stack~
V9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V9/strided_slice/stack_1~
V9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V9/strided_slice/stack_2ô
V9/strided_sliceStridedSliceV9/Shape:output:0V9/strided_slice/stack:output:0!V9/strided_slice/stack_1:output:0!V9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V9/strided_slicej
V9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V9/Reshape/shape/1
V9/Reshape/shapePackV9/strided_slice:output:0V9/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V9/Reshape/shape}

V9/ReshapeReshapefeatures_29V9/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V9/Reshapee
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
concat/axisü
concatConcatV2Amount/Reshape:output:0Time/Reshape:output:0V1/Reshape:output:0V10/Reshape:output:0V11/Reshape:output:0V12/Reshape:output:0V13/Reshape:output:0V14/Reshape:output:0V15/Reshape:output:0V16/Reshape:output:0V17/Reshape:output:0V18/Reshape:output:0V19/Reshape:output:0V2/Reshape:output:0V20/Reshape:output:0V21/Reshape:output:0V22/Reshape:output:0V23/Reshape:output:0V24/Reshape:output:0V25/Reshape:output:0V26/Reshape:output:0V27/Reshape:output:0V28/Reshape:output:0V3/Reshape:output:0V4/Reshape:output:0V5/Reshape:output:0V6/Reshape:output:0V7/Reshape:output:0V8/Reshape:output:0V9/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ï
_input_shapes½
º:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:Q	M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:Q
M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features
±5
Å	
__inference__traced_save_84238
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

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_c6397e77ddff4736818f1cb77c1454d8/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameñ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueù
Bö
B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¶
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÒ	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableopAsavev2_sgd_batch_normalization_gamma_momentum_read_readvariableop@savev2_sgd_batch_normalization_beta_momentum_read_readvariableop4savev2_sgd_dense_kernel_momentum_read_readvariableop2savev2_sgd_dense_bias_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesv
t: ::::::: : : : : : : : :È:È:È:È::::: 2(
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
:È:!

_output_shapes	
:È:!

_output_shapes	
:È:!

_output_shapes	
:È: 
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
«
ð
I__inference_dense_features_layer_call_and_return_conditional_losses_83675
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
identity[
Amount/ShapeShapefeatures_amount*
T0*
_output_shapes
:2
Amount/Shape
Amount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Amount/strided_slice/stack
Amount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Amount/strided_slice/stack_1
Amount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Amount/strided_slice/stack_2
Amount/strided_sliceStridedSliceAmount/Shape:output:0#Amount/strided_slice/stack:output:0%Amount/strided_slice/stack_1:output:0%Amount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Amount/strided_slicer
Amount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Amount/Reshape/shape/1¢
Amount/Reshape/shapePackAmount/strided_slice:output:0Amount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Amount/Reshape/shape
Amount/ReshapeReshapefeatures_amountAmount/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Amount/ReshapeU

Time/ShapeShapefeatures_time*
T0*
_output_shapes
:2

Time/Shape~
Time/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Time/strided_slice/stack
Time/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Time/strided_slice/stack_1
Time/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Time/strided_slice/stack_2
Time/strided_sliceStridedSliceTime/Shape:output:0!Time/strided_slice/stack:output:0#Time/strided_slice/stack_1:output:0#Time/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Time/strided_slicen
Time/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Time/Reshape/shape/1
Time/Reshape/shapePackTime/strided_slice:output:0Time/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Time/Reshape/shape
Time/ReshapeReshapefeatures_timeTime/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Time/ReshapeO
V1/ShapeShapefeatures_v1*
T0*
_output_shapes
:2

V1/Shapez
V1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V1/strided_slice/stack~
V1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V1/strided_slice/stack_1~
V1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V1/strided_slice/stack_2ô
V1/strided_sliceStridedSliceV1/Shape:output:0V1/strided_slice/stack:output:0!V1/strided_slice/stack_1:output:0!V1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V1/strided_slicej
V1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V1/Reshape/shape/1
V1/Reshape/shapePackV1/strided_slice:output:0V1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V1/Reshape/shape}

V1/ReshapeReshapefeatures_v1V1/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V1/ReshapeR
	V10/ShapeShapefeatures_v10*
T0*
_output_shapes
:2
	V10/Shape|
V10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V10/strided_slice/stack
V10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V10/strided_slice/stack_1
V10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V10/strided_slice/stack_2ú
V10/strided_sliceStridedSliceV10/Shape:output:0 V10/strided_slice/stack:output:0"V10/strided_slice/stack_1:output:0"V10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V10/strided_slicel
V10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V10/Reshape/shape/1
V10/Reshape/shapePackV10/strided_slice:output:0V10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V10/Reshape/shape
V10/ReshapeReshapefeatures_v10V10/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V10/ReshapeR
	V11/ShapeShapefeatures_v11*
T0*
_output_shapes
:2
	V11/Shape|
V11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V11/strided_slice/stack
V11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V11/strided_slice/stack_1
V11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V11/strided_slice/stack_2ú
V11/strided_sliceStridedSliceV11/Shape:output:0 V11/strided_slice/stack:output:0"V11/strided_slice/stack_1:output:0"V11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V11/strided_slicel
V11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V11/Reshape/shape/1
V11/Reshape/shapePackV11/strided_slice:output:0V11/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V11/Reshape/shape
V11/ReshapeReshapefeatures_v11V11/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V11/ReshapeR
	V12/ShapeShapefeatures_v12*
T0*
_output_shapes
:2
	V12/Shape|
V12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V12/strided_slice/stack
V12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V12/strided_slice/stack_1
V12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V12/strided_slice/stack_2ú
V12/strided_sliceStridedSliceV12/Shape:output:0 V12/strided_slice/stack:output:0"V12/strided_slice/stack_1:output:0"V12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V12/strided_slicel
V12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V12/Reshape/shape/1
V12/Reshape/shapePackV12/strided_slice:output:0V12/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V12/Reshape/shape
V12/ReshapeReshapefeatures_v12V12/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V12/ReshapeR
	V13/ShapeShapefeatures_v13*
T0*
_output_shapes
:2
	V13/Shape|
V13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V13/strided_slice/stack
V13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V13/strided_slice/stack_1
V13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V13/strided_slice/stack_2ú
V13/strided_sliceStridedSliceV13/Shape:output:0 V13/strided_slice/stack:output:0"V13/strided_slice/stack_1:output:0"V13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V13/strided_slicel
V13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V13/Reshape/shape/1
V13/Reshape/shapePackV13/strided_slice:output:0V13/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V13/Reshape/shape
V13/ReshapeReshapefeatures_v13V13/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V13/ReshapeR
	V14/ShapeShapefeatures_v14*
T0*
_output_shapes
:2
	V14/Shape|
V14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V14/strided_slice/stack
V14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V14/strided_slice/stack_1
V14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V14/strided_slice/stack_2ú
V14/strided_sliceStridedSliceV14/Shape:output:0 V14/strided_slice/stack:output:0"V14/strided_slice/stack_1:output:0"V14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V14/strided_slicel
V14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V14/Reshape/shape/1
V14/Reshape/shapePackV14/strided_slice:output:0V14/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V14/Reshape/shape
V14/ReshapeReshapefeatures_v14V14/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V14/ReshapeR
	V15/ShapeShapefeatures_v15*
T0*
_output_shapes
:2
	V15/Shape|
V15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V15/strided_slice/stack
V15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V15/strided_slice/stack_1
V15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V15/strided_slice/stack_2ú
V15/strided_sliceStridedSliceV15/Shape:output:0 V15/strided_slice/stack:output:0"V15/strided_slice/stack_1:output:0"V15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V15/strided_slicel
V15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V15/Reshape/shape/1
V15/Reshape/shapePackV15/strided_slice:output:0V15/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V15/Reshape/shape
V15/ReshapeReshapefeatures_v15V15/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V15/ReshapeR
	V16/ShapeShapefeatures_v16*
T0*
_output_shapes
:2
	V16/Shape|
V16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V16/strided_slice/stack
V16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V16/strided_slice/stack_1
V16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V16/strided_slice/stack_2ú
V16/strided_sliceStridedSliceV16/Shape:output:0 V16/strided_slice/stack:output:0"V16/strided_slice/stack_1:output:0"V16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V16/strided_slicel
V16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V16/Reshape/shape/1
V16/Reshape/shapePackV16/strided_slice:output:0V16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V16/Reshape/shape
V16/ReshapeReshapefeatures_v16V16/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V16/ReshapeR
	V17/ShapeShapefeatures_v17*
T0*
_output_shapes
:2
	V17/Shape|
V17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V17/strided_slice/stack
V17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V17/strided_slice/stack_1
V17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V17/strided_slice/stack_2ú
V17/strided_sliceStridedSliceV17/Shape:output:0 V17/strided_slice/stack:output:0"V17/strided_slice/stack_1:output:0"V17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V17/strided_slicel
V17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V17/Reshape/shape/1
V17/Reshape/shapePackV17/strided_slice:output:0V17/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V17/Reshape/shape
V17/ReshapeReshapefeatures_v17V17/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V17/ReshapeR
	V18/ShapeShapefeatures_v18*
T0*
_output_shapes
:2
	V18/Shape|
V18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V18/strided_slice/stack
V18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V18/strided_slice/stack_1
V18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V18/strided_slice/stack_2ú
V18/strided_sliceStridedSliceV18/Shape:output:0 V18/strided_slice/stack:output:0"V18/strided_slice/stack_1:output:0"V18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V18/strided_slicel
V18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V18/Reshape/shape/1
V18/Reshape/shapePackV18/strided_slice:output:0V18/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V18/Reshape/shape
V18/ReshapeReshapefeatures_v18V18/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V18/ReshapeR
	V19/ShapeShapefeatures_v19*
T0*
_output_shapes
:2
	V19/Shape|
V19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V19/strided_slice/stack
V19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V19/strided_slice/stack_1
V19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V19/strided_slice/stack_2ú
V19/strided_sliceStridedSliceV19/Shape:output:0 V19/strided_slice/stack:output:0"V19/strided_slice/stack_1:output:0"V19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V19/strided_slicel
V19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V19/Reshape/shape/1
V19/Reshape/shapePackV19/strided_slice:output:0V19/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V19/Reshape/shape
V19/ReshapeReshapefeatures_v19V19/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V19/ReshapeO
V2/ShapeShapefeatures_v2*
T0*
_output_shapes
:2

V2/Shapez
V2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V2/strided_slice/stack~
V2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V2/strided_slice/stack_1~
V2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V2/strided_slice/stack_2ô
V2/strided_sliceStridedSliceV2/Shape:output:0V2/strided_slice/stack:output:0!V2/strided_slice/stack_1:output:0!V2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V2/strided_slicej
V2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V2/Reshape/shape/1
V2/Reshape/shapePackV2/strided_slice:output:0V2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V2/Reshape/shape}

V2/ReshapeReshapefeatures_v2V2/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V2/ReshapeR
	V20/ShapeShapefeatures_v20*
T0*
_output_shapes
:2
	V20/Shape|
V20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V20/strided_slice/stack
V20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V20/strided_slice/stack_1
V20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V20/strided_slice/stack_2ú
V20/strided_sliceStridedSliceV20/Shape:output:0 V20/strided_slice/stack:output:0"V20/strided_slice/stack_1:output:0"V20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V20/strided_slicel
V20/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V20/Reshape/shape/1
V20/Reshape/shapePackV20/strided_slice:output:0V20/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V20/Reshape/shape
V20/ReshapeReshapefeatures_v20V20/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V20/ReshapeR
	V21/ShapeShapefeatures_v21*
T0*
_output_shapes
:2
	V21/Shape|
V21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V21/strided_slice/stack
V21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V21/strided_slice/stack_1
V21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V21/strided_slice/stack_2ú
V21/strided_sliceStridedSliceV21/Shape:output:0 V21/strided_slice/stack:output:0"V21/strided_slice/stack_1:output:0"V21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V21/strided_slicel
V21/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V21/Reshape/shape/1
V21/Reshape/shapePackV21/strided_slice:output:0V21/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V21/Reshape/shape
V21/ReshapeReshapefeatures_v21V21/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V21/ReshapeR
	V22/ShapeShapefeatures_v22*
T0*
_output_shapes
:2
	V22/Shape|
V22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V22/strided_slice/stack
V22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V22/strided_slice/stack_1
V22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V22/strided_slice/stack_2ú
V22/strided_sliceStridedSliceV22/Shape:output:0 V22/strided_slice/stack:output:0"V22/strided_slice/stack_1:output:0"V22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V22/strided_slicel
V22/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V22/Reshape/shape/1
V22/Reshape/shapePackV22/strided_slice:output:0V22/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V22/Reshape/shape
V22/ReshapeReshapefeatures_v22V22/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V22/ReshapeR
	V23/ShapeShapefeatures_v23*
T0*
_output_shapes
:2
	V23/Shape|
V23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V23/strided_slice/stack
V23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V23/strided_slice/stack_1
V23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V23/strided_slice/stack_2ú
V23/strided_sliceStridedSliceV23/Shape:output:0 V23/strided_slice/stack:output:0"V23/strided_slice/stack_1:output:0"V23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V23/strided_slicel
V23/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V23/Reshape/shape/1
V23/Reshape/shapePackV23/strided_slice:output:0V23/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V23/Reshape/shape
V23/ReshapeReshapefeatures_v23V23/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V23/ReshapeR
	V24/ShapeShapefeatures_v24*
T0*
_output_shapes
:2
	V24/Shape|
V24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V24/strided_slice/stack
V24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V24/strided_slice/stack_1
V24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V24/strided_slice/stack_2ú
V24/strided_sliceStridedSliceV24/Shape:output:0 V24/strided_slice/stack:output:0"V24/strided_slice/stack_1:output:0"V24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V24/strided_slicel
V24/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V24/Reshape/shape/1
V24/Reshape/shapePackV24/strided_slice:output:0V24/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V24/Reshape/shape
V24/ReshapeReshapefeatures_v24V24/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V24/ReshapeR
	V25/ShapeShapefeatures_v25*
T0*
_output_shapes
:2
	V25/Shape|
V25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V25/strided_slice/stack
V25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V25/strided_slice/stack_1
V25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V25/strided_slice/stack_2ú
V25/strided_sliceStridedSliceV25/Shape:output:0 V25/strided_slice/stack:output:0"V25/strided_slice/stack_1:output:0"V25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V25/strided_slicel
V25/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V25/Reshape/shape/1
V25/Reshape/shapePackV25/strided_slice:output:0V25/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V25/Reshape/shape
V25/ReshapeReshapefeatures_v25V25/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V25/ReshapeR
	V26/ShapeShapefeatures_v26*
T0*
_output_shapes
:2
	V26/Shape|
V26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V26/strided_slice/stack
V26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V26/strided_slice/stack_1
V26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V26/strided_slice/stack_2ú
V26/strided_sliceStridedSliceV26/Shape:output:0 V26/strided_slice/stack:output:0"V26/strided_slice/stack_1:output:0"V26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V26/strided_slicel
V26/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V26/Reshape/shape/1
V26/Reshape/shapePackV26/strided_slice:output:0V26/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V26/Reshape/shape
V26/ReshapeReshapefeatures_v26V26/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V26/ReshapeR
	V27/ShapeShapefeatures_v27*
T0*
_output_shapes
:2
	V27/Shape|
V27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V27/strided_slice/stack
V27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V27/strided_slice/stack_1
V27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V27/strided_slice/stack_2ú
V27/strided_sliceStridedSliceV27/Shape:output:0 V27/strided_slice/stack:output:0"V27/strided_slice/stack_1:output:0"V27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V27/strided_slicel
V27/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V27/Reshape/shape/1
V27/Reshape/shapePackV27/strided_slice:output:0V27/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V27/Reshape/shape
V27/ReshapeReshapefeatures_v27V27/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V27/ReshapeR
	V28/ShapeShapefeatures_v28*
T0*
_output_shapes
:2
	V28/Shape|
V28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V28/strided_slice/stack
V28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V28/strided_slice/stack_1
V28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V28/strided_slice/stack_2ú
V28/strided_sliceStridedSliceV28/Shape:output:0 V28/strided_slice/stack:output:0"V28/strided_slice/stack_1:output:0"V28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V28/strided_slicel
V28/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V28/Reshape/shape/1
V28/Reshape/shapePackV28/strided_slice:output:0V28/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V28/Reshape/shape
V28/ReshapeReshapefeatures_v28V28/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V28/ReshapeO
V3/ShapeShapefeatures_v3*
T0*
_output_shapes
:2

V3/Shapez
V3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V3/strided_slice/stack~
V3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V3/strided_slice/stack_1~
V3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V3/strided_slice/stack_2ô
V3/strided_sliceStridedSliceV3/Shape:output:0V3/strided_slice/stack:output:0!V3/strided_slice/stack_1:output:0!V3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V3/strided_slicej
V3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V3/Reshape/shape/1
V3/Reshape/shapePackV3/strided_slice:output:0V3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V3/Reshape/shape}

V3/ReshapeReshapefeatures_v3V3/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V3/ReshapeO
V4/ShapeShapefeatures_v4*
T0*
_output_shapes
:2

V4/Shapez
V4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V4/strided_slice/stack~
V4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V4/strided_slice/stack_1~
V4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V4/strided_slice/stack_2ô
V4/strided_sliceStridedSliceV4/Shape:output:0V4/strided_slice/stack:output:0!V4/strided_slice/stack_1:output:0!V4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V4/strided_slicej
V4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V4/Reshape/shape/1
V4/Reshape/shapePackV4/strided_slice:output:0V4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V4/Reshape/shape}

V4/ReshapeReshapefeatures_v4V4/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V4/ReshapeO
V5/ShapeShapefeatures_v5*
T0*
_output_shapes
:2

V5/Shapez
V5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V5/strided_slice/stack~
V5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V5/strided_slice/stack_1~
V5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V5/strided_slice/stack_2ô
V5/strided_sliceStridedSliceV5/Shape:output:0V5/strided_slice/stack:output:0!V5/strided_slice/stack_1:output:0!V5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V5/strided_slicej
V5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V5/Reshape/shape/1
V5/Reshape/shapePackV5/strided_slice:output:0V5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V5/Reshape/shape}

V5/ReshapeReshapefeatures_v5V5/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V5/ReshapeO
V6/ShapeShapefeatures_v6*
T0*
_output_shapes
:2

V6/Shapez
V6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V6/strided_slice/stack~
V6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V6/strided_slice/stack_1~
V6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V6/strided_slice/stack_2ô
V6/strided_sliceStridedSliceV6/Shape:output:0V6/strided_slice/stack:output:0!V6/strided_slice/stack_1:output:0!V6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V6/strided_slicej
V6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V6/Reshape/shape/1
V6/Reshape/shapePackV6/strided_slice:output:0V6/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V6/Reshape/shape}

V6/ReshapeReshapefeatures_v6V6/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V6/ReshapeO
V7/ShapeShapefeatures_v7*
T0*
_output_shapes
:2

V7/Shapez
V7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V7/strided_slice/stack~
V7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V7/strided_slice/stack_1~
V7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V7/strided_slice/stack_2ô
V7/strided_sliceStridedSliceV7/Shape:output:0V7/strided_slice/stack:output:0!V7/strided_slice/stack_1:output:0!V7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V7/strided_slicej
V7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V7/Reshape/shape/1
V7/Reshape/shapePackV7/strided_slice:output:0V7/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V7/Reshape/shape}

V7/ReshapeReshapefeatures_v7V7/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V7/ReshapeO
V8/ShapeShapefeatures_v8*
T0*
_output_shapes
:2

V8/Shapez
V8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V8/strided_slice/stack~
V8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V8/strided_slice/stack_1~
V8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V8/strided_slice/stack_2ô
V8/strided_sliceStridedSliceV8/Shape:output:0V8/strided_slice/stack:output:0!V8/strided_slice/stack_1:output:0!V8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V8/strided_slicej
V8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V8/Reshape/shape/1
V8/Reshape/shapePackV8/strided_slice:output:0V8/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V8/Reshape/shape}

V8/ReshapeReshapefeatures_v8V8/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V8/ReshapeO
V9/ShapeShapefeatures_v9*
T0*
_output_shapes
:2

V9/Shapez
V9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V9/strided_slice/stack~
V9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V9/strided_slice/stack_1~
V9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V9/strided_slice/stack_2ô
V9/strided_sliceStridedSliceV9/Shape:output:0V9/strided_slice/stack:output:0!V9/strided_slice/stack_1:output:0!V9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V9/strided_slicej
V9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V9/Reshape/shape/1
V9/Reshape/shapePackV9/strided_slice:output:0V9/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V9/Reshape/shape}

V9/ReshapeReshapefeatures_v9V9/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V9/Reshapee
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
concat/axisü
concatConcatV2Amount/Reshape:output:0Time/Reshape:output:0V1/Reshape:output:0V10/Reshape:output:0V11/Reshape:output:0V12/Reshape:output:0V13/Reshape:output:0V14/Reshape:output:0V15/Reshape:output:0V16/Reshape:output:0V17/Reshape:output:0V18/Reshape:output:0V19/Reshape:output:0V2/Reshape:output:0V20/Reshape:output:0V21/Reshape:output:0V22/Reshape:output:0V23/Reshape:output:0V24/Reshape:output:0V25/Reshape:output:0V26/Reshape:output:0V27/Reshape:output:0V28/Reshape:output:0V3/Reshape:output:0V4/Reshape:output:0V5/Reshape:output:0V6/Reshape:output:0V7/Reshape:output:0V8/Reshape:output:0V9/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ï
_input_shapes½
º:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:X T
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_namefeatures/Amount:VR
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namefeatures/Time:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V1:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V10:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V11:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V12:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V13:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V14:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V15:U	Q
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V16:U
Q
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V17:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V18:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V19:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V2:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V20:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V21:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V22:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V23:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V24:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V25:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V26:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V27:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V28:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V3:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V4:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V5:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V6:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V7:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V8:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V9


N__inference_batch_normalization_layer_call_and_return_conditional_losses_84074

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
)
Å
N__inference_batch_normalization_layer_call_and_return_conditional_losses_81644

inputs
assignmovingavg_81619
assignmovingavg_1_81625)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*(
_class
loc:@AssignMovingAvg/81619*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_81619*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpÂ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*(
_class
loc:@AssignMovingAvg/81619*
_output_shapes
:2
AssignMovingAvg/sub¹
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*(
_class
loc:@AssignMovingAvg/81619*
_output_shapes
:2
AssignMovingAvg/mulÿ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_81619AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/81619*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp£
AssignMovingAvg_1/decayConst**
_class 
loc:@AssignMovingAvg_1/81625*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_81625*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÌ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/81625*
_output_shapes
:2
AssignMovingAvg_1/subÃ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/81625*
_output_shapes
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_81625AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/81625*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1µ
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©ä
Ó
 __inference__wrapped_model_81548

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
v9F
Bfunctional_1_batch_normalization_batchnorm_readvariableop_resourceJ
Ffunctional_1_batch_normalization_batchnorm_mul_readvariableop_resourceH
Dfunctional_1_batch_normalization_batchnorm_readvariableop_1_resourceH
Dfunctional_1_batch_normalization_batchnorm_readvariableop_2_resource5
1functional_1_dense_matmul_readvariableop_resource6
2functional_1_dense_biasadd_readvariableop_resource
identity
(functional_1/dense_features/Amount/ShapeShapeamount*
T0*
_output_shapes
:2*
(functional_1/dense_features/Amount/Shapeº
6functional_1/dense_features/Amount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6functional_1/dense_features/Amount/strided_slice/stack¾
8functional_1/dense_features/Amount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8functional_1/dense_features/Amount/strided_slice/stack_1¾
8functional_1/dense_features/Amount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8functional_1/dense_features/Amount/strided_slice/stack_2´
0functional_1/dense_features/Amount/strided_sliceStridedSlice1functional_1/dense_features/Amount/Shape:output:0?functional_1/dense_features/Amount/strided_slice/stack:output:0Afunctional_1/dense_features/Amount/strided_slice/stack_1:output:0Afunctional_1/dense_features/Amount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0functional_1/dense_features/Amount/strided_sliceª
2functional_1/dense_features/Amount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :24
2functional_1/dense_features/Amount/Reshape/shape/1
0functional_1/dense_features/Amount/Reshape/shapePack9functional_1/dense_features/Amount/strided_slice:output:0;functional_1/dense_features/Amount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:22
0functional_1/dense_features/Amount/Reshape/shapeØ
*functional_1/dense_features/Amount/ReshapeReshapeamount9functional_1/dense_features/Amount/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*functional_1/dense_features/Amount/Reshape
&functional_1/dense_features/Time/ShapeShapetime*
T0*
_output_shapes
:2(
&functional_1/dense_features/Time/Shape¶
4functional_1/dense_features/Time/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4functional_1/dense_features/Time/strided_slice/stackº
6functional_1/dense_features/Time/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6functional_1/dense_features/Time/strided_slice/stack_1º
6functional_1/dense_features/Time/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6functional_1/dense_features/Time/strided_slice/stack_2¨
.functional_1/dense_features/Time/strided_sliceStridedSlice/functional_1/dense_features/Time/Shape:output:0=functional_1/dense_features/Time/strided_slice/stack:output:0?functional_1/dense_features/Time/strided_slice/stack_1:output:0?functional_1/dense_features/Time/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.functional_1/dense_features/Time/strided_slice¦
0functional_1/dense_features/Time/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0functional_1/dense_features/Time/Reshape/shape/1
.functional_1/dense_features/Time/Reshape/shapePack7functional_1/dense_features/Time/strided_slice:output:09functional_1/dense_features/Time/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:20
.functional_1/dense_features/Time/Reshape/shapeÐ
(functional_1/dense_features/Time/ReshapeReshapetime7functional_1/dense_features/Time/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(functional_1/dense_features/Time/Reshape~
$functional_1/dense_features/V1/ShapeShapev1*
T0*
_output_shapes
:2&
$functional_1/dense_features/V1/Shape²
2functional_1/dense_features/V1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2functional_1/dense_features/V1/strided_slice/stack¶
4functional_1/dense_features/V1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_1/dense_features/V1/strided_slice/stack_1¶
4functional_1/dense_features/V1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_1/dense_features/V1/strided_slice/stack_2
,functional_1/dense_features/V1/strided_sliceStridedSlice-functional_1/dense_features/V1/Shape:output:0;functional_1/dense_features/V1/strided_slice/stack:output:0=functional_1/dense_features/V1/strided_slice/stack_1:output:0=functional_1/dense_features/V1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,functional_1/dense_features/V1/strided_slice¢
.functional_1/dense_features/V1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.functional_1/dense_features/V1/Reshape/shape/1
,functional_1/dense_features/V1/Reshape/shapePack5functional_1/dense_features/V1/strided_slice:output:07functional_1/dense_features/V1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,functional_1/dense_features/V1/Reshape/shapeÈ
&functional_1/dense_features/V1/ReshapeReshapev15functional_1/dense_features/V1/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&functional_1/dense_features/V1/Reshape
%functional_1/dense_features/V10/ShapeShapev10*
T0*
_output_shapes
:2'
%functional_1/dense_features/V10/Shape´
3functional_1/dense_features/V10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_1/dense_features/V10/strided_slice/stack¸
5functional_1/dense_features/V10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V10/strided_slice/stack_1¸
5functional_1/dense_features/V10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V10/strided_slice/stack_2¢
-functional_1/dense_features/V10/strided_sliceStridedSlice.functional_1/dense_features/V10/Shape:output:0<functional_1/dense_features/V10/strided_slice/stack:output:0>functional_1/dense_features/V10/strided_slice/stack_1:output:0>functional_1/dense_features/V10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_1/dense_features/V10/strided_slice¤
/functional_1/dense_features/V10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/dense_features/V10/Reshape/shape/1
-functional_1/dense_features/V10/Reshape/shapePack6functional_1/dense_features/V10/strided_slice:output:08functional_1/dense_features/V10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-functional_1/dense_features/V10/Reshape/shapeÌ
'functional_1/dense_features/V10/ReshapeReshapev106functional_1/dense_features/V10/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_1/dense_features/V10/Reshape
%functional_1/dense_features/V11/ShapeShapev11*
T0*
_output_shapes
:2'
%functional_1/dense_features/V11/Shape´
3functional_1/dense_features/V11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_1/dense_features/V11/strided_slice/stack¸
5functional_1/dense_features/V11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V11/strided_slice/stack_1¸
5functional_1/dense_features/V11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V11/strided_slice/stack_2¢
-functional_1/dense_features/V11/strided_sliceStridedSlice.functional_1/dense_features/V11/Shape:output:0<functional_1/dense_features/V11/strided_slice/stack:output:0>functional_1/dense_features/V11/strided_slice/stack_1:output:0>functional_1/dense_features/V11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_1/dense_features/V11/strided_slice¤
/functional_1/dense_features/V11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/dense_features/V11/Reshape/shape/1
-functional_1/dense_features/V11/Reshape/shapePack6functional_1/dense_features/V11/strided_slice:output:08functional_1/dense_features/V11/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-functional_1/dense_features/V11/Reshape/shapeÌ
'functional_1/dense_features/V11/ReshapeReshapev116functional_1/dense_features/V11/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_1/dense_features/V11/Reshape
%functional_1/dense_features/V12/ShapeShapev12*
T0*
_output_shapes
:2'
%functional_1/dense_features/V12/Shape´
3functional_1/dense_features/V12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_1/dense_features/V12/strided_slice/stack¸
5functional_1/dense_features/V12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V12/strided_slice/stack_1¸
5functional_1/dense_features/V12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V12/strided_slice/stack_2¢
-functional_1/dense_features/V12/strided_sliceStridedSlice.functional_1/dense_features/V12/Shape:output:0<functional_1/dense_features/V12/strided_slice/stack:output:0>functional_1/dense_features/V12/strided_slice/stack_1:output:0>functional_1/dense_features/V12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_1/dense_features/V12/strided_slice¤
/functional_1/dense_features/V12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/dense_features/V12/Reshape/shape/1
-functional_1/dense_features/V12/Reshape/shapePack6functional_1/dense_features/V12/strided_slice:output:08functional_1/dense_features/V12/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-functional_1/dense_features/V12/Reshape/shapeÌ
'functional_1/dense_features/V12/ReshapeReshapev126functional_1/dense_features/V12/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_1/dense_features/V12/Reshape
%functional_1/dense_features/V13/ShapeShapev13*
T0*
_output_shapes
:2'
%functional_1/dense_features/V13/Shape´
3functional_1/dense_features/V13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_1/dense_features/V13/strided_slice/stack¸
5functional_1/dense_features/V13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V13/strided_slice/stack_1¸
5functional_1/dense_features/V13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V13/strided_slice/stack_2¢
-functional_1/dense_features/V13/strided_sliceStridedSlice.functional_1/dense_features/V13/Shape:output:0<functional_1/dense_features/V13/strided_slice/stack:output:0>functional_1/dense_features/V13/strided_slice/stack_1:output:0>functional_1/dense_features/V13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_1/dense_features/V13/strided_slice¤
/functional_1/dense_features/V13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/dense_features/V13/Reshape/shape/1
-functional_1/dense_features/V13/Reshape/shapePack6functional_1/dense_features/V13/strided_slice:output:08functional_1/dense_features/V13/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-functional_1/dense_features/V13/Reshape/shapeÌ
'functional_1/dense_features/V13/ReshapeReshapev136functional_1/dense_features/V13/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_1/dense_features/V13/Reshape
%functional_1/dense_features/V14/ShapeShapev14*
T0*
_output_shapes
:2'
%functional_1/dense_features/V14/Shape´
3functional_1/dense_features/V14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_1/dense_features/V14/strided_slice/stack¸
5functional_1/dense_features/V14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V14/strided_slice/stack_1¸
5functional_1/dense_features/V14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V14/strided_slice/stack_2¢
-functional_1/dense_features/V14/strided_sliceStridedSlice.functional_1/dense_features/V14/Shape:output:0<functional_1/dense_features/V14/strided_slice/stack:output:0>functional_1/dense_features/V14/strided_slice/stack_1:output:0>functional_1/dense_features/V14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_1/dense_features/V14/strided_slice¤
/functional_1/dense_features/V14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/dense_features/V14/Reshape/shape/1
-functional_1/dense_features/V14/Reshape/shapePack6functional_1/dense_features/V14/strided_slice:output:08functional_1/dense_features/V14/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-functional_1/dense_features/V14/Reshape/shapeÌ
'functional_1/dense_features/V14/ReshapeReshapev146functional_1/dense_features/V14/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_1/dense_features/V14/Reshape
%functional_1/dense_features/V15/ShapeShapev15*
T0*
_output_shapes
:2'
%functional_1/dense_features/V15/Shape´
3functional_1/dense_features/V15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_1/dense_features/V15/strided_slice/stack¸
5functional_1/dense_features/V15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V15/strided_slice/stack_1¸
5functional_1/dense_features/V15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V15/strided_slice/stack_2¢
-functional_1/dense_features/V15/strided_sliceStridedSlice.functional_1/dense_features/V15/Shape:output:0<functional_1/dense_features/V15/strided_slice/stack:output:0>functional_1/dense_features/V15/strided_slice/stack_1:output:0>functional_1/dense_features/V15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_1/dense_features/V15/strided_slice¤
/functional_1/dense_features/V15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/dense_features/V15/Reshape/shape/1
-functional_1/dense_features/V15/Reshape/shapePack6functional_1/dense_features/V15/strided_slice:output:08functional_1/dense_features/V15/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-functional_1/dense_features/V15/Reshape/shapeÌ
'functional_1/dense_features/V15/ReshapeReshapev156functional_1/dense_features/V15/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_1/dense_features/V15/Reshape
%functional_1/dense_features/V16/ShapeShapev16*
T0*
_output_shapes
:2'
%functional_1/dense_features/V16/Shape´
3functional_1/dense_features/V16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_1/dense_features/V16/strided_slice/stack¸
5functional_1/dense_features/V16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V16/strided_slice/stack_1¸
5functional_1/dense_features/V16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V16/strided_slice/stack_2¢
-functional_1/dense_features/V16/strided_sliceStridedSlice.functional_1/dense_features/V16/Shape:output:0<functional_1/dense_features/V16/strided_slice/stack:output:0>functional_1/dense_features/V16/strided_slice/stack_1:output:0>functional_1/dense_features/V16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_1/dense_features/V16/strided_slice¤
/functional_1/dense_features/V16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/dense_features/V16/Reshape/shape/1
-functional_1/dense_features/V16/Reshape/shapePack6functional_1/dense_features/V16/strided_slice:output:08functional_1/dense_features/V16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-functional_1/dense_features/V16/Reshape/shapeÌ
'functional_1/dense_features/V16/ReshapeReshapev166functional_1/dense_features/V16/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_1/dense_features/V16/Reshape
%functional_1/dense_features/V17/ShapeShapev17*
T0*
_output_shapes
:2'
%functional_1/dense_features/V17/Shape´
3functional_1/dense_features/V17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_1/dense_features/V17/strided_slice/stack¸
5functional_1/dense_features/V17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V17/strided_slice/stack_1¸
5functional_1/dense_features/V17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V17/strided_slice/stack_2¢
-functional_1/dense_features/V17/strided_sliceStridedSlice.functional_1/dense_features/V17/Shape:output:0<functional_1/dense_features/V17/strided_slice/stack:output:0>functional_1/dense_features/V17/strided_slice/stack_1:output:0>functional_1/dense_features/V17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_1/dense_features/V17/strided_slice¤
/functional_1/dense_features/V17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/dense_features/V17/Reshape/shape/1
-functional_1/dense_features/V17/Reshape/shapePack6functional_1/dense_features/V17/strided_slice:output:08functional_1/dense_features/V17/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-functional_1/dense_features/V17/Reshape/shapeÌ
'functional_1/dense_features/V17/ReshapeReshapev176functional_1/dense_features/V17/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_1/dense_features/V17/Reshape
%functional_1/dense_features/V18/ShapeShapev18*
T0*
_output_shapes
:2'
%functional_1/dense_features/V18/Shape´
3functional_1/dense_features/V18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_1/dense_features/V18/strided_slice/stack¸
5functional_1/dense_features/V18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V18/strided_slice/stack_1¸
5functional_1/dense_features/V18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V18/strided_slice/stack_2¢
-functional_1/dense_features/V18/strided_sliceStridedSlice.functional_1/dense_features/V18/Shape:output:0<functional_1/dense_features/V18/strided_slice/stack:output:0>functional_1/dense_features/V18/strided_slice/stack_1:output:0>functional_1/dense_features/V18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_1/dense_features/V18/strided_slice¤
/functional_1/dense_features/V18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/dense_features/V18/Reshape/shape/1
-functional_1/dense_features/V18/Reshape/shapePack6functional_1/dense_features/V18/strided_slice:output:08functional_1/dense_features/V18/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-functional_1/dense_features/V18/Reshape/shapeÌ
'functional_1/dense_features/V18/ReshapeReshapev186functional_1/dense_features/V18/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_1/dense_features/V18/Reshape
%functional_1/dense_features/V19/ShapeShapev19*
T0*
_output_shapes
:2'
%functional_1/dense_features/V19/Shape´
3functional_1/dense_features/V19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_1/dense_features/V19/strided_slice/stack¸
5functional_1/dense_features/V19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V19/strided_slice/stack_1¸
5functional_1/dense_features/V19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V19/strided_slice/stack_2¢
-functional_1/dense_features/V19/strided_sliceStridedSlice.functional_1/dense_features/V19/Shape:output:0<functional_1/dense_features/V19/strided_slice/stack:output:0>functional_1/dense_features/V19/strided_slice/stack_1:output:0>functional_1/dense_features/V19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_1/dense_features/V19/strided_slice¤
/functional_1/dense_features/V19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/dense_features/V19/Reshape/shape/1
-functional_1/dense_features/V19/Reshape/shapePack6functional_1/dense_features/V19/strided_slice:output:08functional_1/dense_features/V19/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-functional_1/dense_features/V19/Reshape/shapeÌ
'functional_1/dense_features/V19/ReshapeReshapev196functional_1/dense_features/V19/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_1/dense_features/V19/Reshape~
$functional_1/dense_features/V2/ShapeShapev2*
T0*
_output_shapes
:2&
$functional_1/dense_features/V2/Shape²
2functional_1/dense_features/V2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2functional_1/dense_features/V2/strided_slice/stack¶
4functional_1/dense_features/V2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_1/dense_features/V2/strided_slice/stack_1¶
4functional_1/dense_features/V2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_1/dense_features/V2/strided_slice/stack_2
,functional_1/dense_features/V2/strided_sliceStridedSlice-functional_1/dense_features/V2/Shape:output:0;functional_1/dense_features/V2/strided_slice/stack:output:0=functional_1/dense_features/V2/strided_slice/stack_1:output:0=functional_1/dense_features/V2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,functional_1/dense_features/V2/strided_slice¢
.functional_1/dense_features/V2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.functional_1/dense_features/V2/Reshape/shape/1
,functional_1/dense_features/V2/Reshape/shapePack5functional_1/dense_features/V2/strided_slice:output:07functional_1/dense_features/V2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,functional_1/dense_features/V2/Reshape/shapeÈ
&functional_1/dense_features/V2/ReshapeReshapev25functional_1/dense_features/V2/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&functional_1/dense_features/V2/Reshape
%functional_1/dense_features/V20/ShapeShapev20*
T0*
_output_shapes
:2'
%functional_1/dense_features/V20/Shape´
3functional_1/dense_features/V20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_1/dense_features/V20/strided_slice/stack¸
5functional_1/dense_features/V20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V20/strided_slice/stack_1¸
5functional_1/dense_features/V20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V20/strided_slice/stack_2¢
-functional_1/dense_features/V20/strided_sliceStridedSlice.functional_1/dense_features/V20/Shape:output:0<functional_1/dense_features/V20/strided_slice/stack:output:0>functional_1/dense_features/V20/strided_slice/stack_1:output:0>functional_1/dense_features/V20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_1/dense_features/V20/strided_slice¤
/functional_1/dense_features/V20/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/dense_features/V20/Reshape/shape/1
-functional_1/dense_features/V20/Reshape/shapePack6functional_1/dense_features/V20/strided_slice:output:08functional_1/dense_features/V20/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-functional_1/dense_features/V20/Reshape/shapeÌ
'functional_1/dense_features/V20/ReshapeReshapev206functional_1/dense_features/V20/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_1/dense_features/V20/Reshape
%functional_1/dense_features/V21/ShapeShapev21*
T0*
_output_shapes
:2'
%functional_1/dense_features/V21/Shape´
3functional_1/dense_features/V21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_1/dense_features/V21/strided_slice/stack¸
5functional_1/dense_features/V21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V21/strided_slice/stack_1¸
5functional_1/dense_features/V21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V21/strided_slice/stack_2¢
-functional_1/dense_features/V21/strided_sliceStridedSlice.functional_1/dense_features/V21/Shape:output:0<functional_1/dense_features/V21/strided_slice/stack:output:0>functional_1/dense_features/V21/strided_slice/stack_1:output:0>functional_1/dense_features/V21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_1/dense_features/V21/strided_slice¤
/functional_1/dense_features/V21/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/dense_features/V21/Reshape/shape/1
-functional_1/dense_features/V21/Reshape/shapePack6functional_1/dense_features/V21/strided_slice:output:08functional_1/dense_features/V21/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-functional_1/dense_features/V21/Reshape/shapeÌ
'functional_1/dense_features/V21/ReshapeReshapev216functional_1/dense_features/V21/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_1/dense_features/V21/Reshape
%functional_1/dense_features/V22/ShapeShapev22*
T0*
_output_shapes
:2'
%functional_1/dense_features/V22/Shape´
3functional_1/dense_features/V22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_1/dense_features/V22/strided_slice/stack¸
5functional_1/dense_features/V22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V22/strided_slice/stack_1¸
5functional_1/dense_features/V22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V22/strided_slice/stack_2¢
-functional_1/dense_features/V22/strided_sliceStridedSlice.functional_1/dense_features/V22/Shape:output:0<functional_1/dense_features/V22/strided_slice/stack:output:0>functional_1/dense_features/V22/strided_slice/stack_1:output:0>functional_1/dense_features/V22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_1/dense_features/V22/strided_slice¤
/functional_1/dense_features/V22/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/dense_features/V22/Reshape/shape/1
-functional_1/dense_features/V22/Reshape/shapePack6functional_1/dense_features/V22/strided_slice:output:08functional_1/dense_features/V22/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-functional_1/dense_features/V22/Reshape/shapeÌ
'functional_1/dense_features/V22/ReshapeReshapev226functional_1/dense_features/V22/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_1/dense_features/V22/Reshape
%functional_1/dense_features/V23/ShapeShapev23*
T0*
_output_shapes
:2'
%functional_1/dense_features/V23/Shape´
3functional_1/dense_features/V23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_1/dense_features/V23/strided_slice/stack¸
5functional_1/dense_features/V23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V23/strided_slice/stack_1¸
5functional_1/dense_features/V23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V23/strided_slice/stack_2¢
-functional_1/dense_features/V23/strided_sliceStridedSlice.functional_1/dense_features/V23/Shape:output:0<functional_1/dense_features/V23/strided_slice/stack:output:0>functional_1/dense_features/V23/strided_slice/stack_1:output:0>functional_1/dense_features/V23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_1/dense_features/V23/strided_slice¤
/functional_1/dense_features/V23/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/dense_features/V23/Reshape/shape/1
-functional_1/dense_features/V23/Reshape/shapePack6functional_1/dense_features/V23/strided_slice:output:08functional_1/dense_features/V23/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-functional_1/dense_features/V23/Reshape/shapeÌ
'functional_1/dense_features/V23/ReshapeReshapev236functional_1/dense_features/V23/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_1/dense_features/V23/Reshape
%functional_1/dense_features/V24/ShapeShapev24*
T0*
_output_shapes
:2'
%functional_1/dense_features/V24/Shape´
3functional_1/dense_features/V24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_1/dense_features/V24/strided_slice/stack¸
5functional_1/dense_features/V24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V24/strided_slice/stack_1¸
5functional_1/dense_features/V24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V24/strided_slice/stack_2¢
-functional_1/dense_features/V24/strided_sliceStridedSlice.functional_1/dense_features/V24/Shape:output:0<functional_1/dense_features/V24/strided_slice/stack:output:0>functional_1/dense_features/V24/strided_slice/stack_1:output:0>functional_1/dense_features/V24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_1/dense_features/V24/strided_slice¤
/functional_1/dense_features/V24/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/dense_features/V24/Reshape/shape/1
-functional_1/dense_features/V24/Reshape/shapePack6functional_1/dense_features/V24/strided_slice:output:08functional_1/dense_features/V24/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-functional_1/dense_features/V24/Reshape/shapeÌ
'functional_1/dense_features/V24/ReshapeReshapev246functional_1/dense_features/V24/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_1/dense_features/V24/Reshape
%functional_1/dense_features/V25/ShapeShapev25*
T0*
_output_shapes
:2'
%functional_1/dense_features/V25/Shape´
3functional_1/dense_features/V25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_1/dense_features/V25/strided_slice/stack¸
5functional_1/dense_features/V25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V25/strided_slice/stack_1¸
5functional_1/dense_features/V25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V25/strided_slice/stack_2¢
-functional_1/dense_features/V25/strided_sliceStridedSlice.functional_1/dense_features/V25/Shape:output:0<functional_1/dense_features/V25/strided_slice/stack:output:0>functional_1/dense_features/V25/strided_slice/stack_1:output:0>functional_1/dense_features/V25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_1/dense_features/V25/strided_slice¤
/functional_1/dense_features/V25/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/dense_features/V25/Reshape/shape/1
-functional_1/dense_features/V25/Reshape/shapePack6functional_1/dense_features/V25/strided_slice:output:08functional_1/dense_features/V25/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-functional_1/dense_features/V25/Reshape/shapeÌ
'functional_1/dense_features/V25/ReshapeReshapev256functional_1/dense_features/V25/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_1/dense_features/V25/Reshape
%functional_1/dense_features/V26/ShapeShapev26*
T0*
_output_shapes
:2'
%functional_1/dense_features/V26/Shape´
3functional_1/dense_features/V26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_1/dense_features/V26/strided_slice/stack¸
5functional_1/dense_features/V26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V26/strided_slice/stack_1¸
5functional_1/dense_features/V26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V26/strided_slice/stack_2¢
-functional_1/dense_features/V26/strided_sliceStridedSlice.functional_1/dense_features/V26/Shape:output:0<functional_1/dense_features/V26/strided_slice/stack:output:0>functional_1/dense_features/V26/strided_slice/stack_1:output:0>functional_1/dense_features/V26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_1/dense_features/V26/strided_slice¤
/functional_1/dense_features/V26/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/dense_features/V26/Reshape/shape/1
-functional_1/dense_features/V26/Reshape/shapePack6functional_1/dense_features/V26/strided_slice:output:08functional_1/dense_features/V26/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-functional_1/dense_features/V26/Reshape/shapeÌ
'functional_1/dense_features/V26/ReshapeReshapev266functional_1/dense_features/V26/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_1/dense_features/V26/Reshape
%functional_1/dense_features/V27/ShapeShapev27*
T0*
_output_shapes
:2'
%functional_1/dense_features/V27/Shape´
3functional_1/dense_features/V27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_1/dense_features/V27/strided_slice/stack¸
5functional_1/dense_features/V27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V27/strided_slice/stack_1¸
5functional_1/dense_features/V27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V27/strided_slice/stack_2¢
-functional_1/dense_features/V27/strided_sliceStridedSlice.functional_1/dense_features/V27/Shape:output:0<functional_1/dense_features/V27/strided_slice/stack:output:0>functional_1/dense_features/V27/strided_slice/stack_1:output:0>functional_1/dense_features/V27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_1/dense_features/V27/strided_slice¤
/functional_1/dense_features/V27/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/dense_features/V27/Reshape/shape/1
-functional_1/dense_features/V27/Reshape/shapePack6functional_1/dense_features/V27/strided_slice:output:08functional_1/dense_features/V27/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-functional_1/dense_features/V27/Reshape/shapeÌ
'functional_1/dense_features/V27/ReshapeReshapev276functional_1/dense_features/V27/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_1/dense_features/V27/Reshape
%functional_1/dense_features/V28/ShapeShapev28*
T0*
_output_shapes
:2'
%functional_1/dense_features/V28/Shape´
3functional_1/dense_features/V28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_1/dense_features/V28/strided_slice/stack¸
5functional_1/dense_features/V28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V28/strided_slice/stack_1¸
5functional_1/dense_features/V28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_1/dense_features/V28/strided_slice/stack_2¢
-functional_1/dense_features/V28/strided_sliceStridedSlice.functional_1/dense_features/V28/Shape:output:0<functional_1/dense_features/V28/strided_slice/stack:output:0>functional_1/dense_features/V28/strided_slice/stack_1:output:0>functional_1/dense_features/V28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_1/dense_features/V28/strided_slice¤
/functional_1/dense_features/V28/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/dense_features/V28/Reshape/shape/1
-functional_1/dense_features/V28/Reshape/shapePack6functional_1/dense_features/V28/strided_slice:output:08functional_1/dense_features/V28/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-functional_1/dense_features/V28/Reshape/shapeÌ
'functional_1/dense_features/V28/ReshapeReshapev286functional_1/dense_features/V28/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_1/dense_features/V28/Reshape~
$functional_1/dense_features/V3/ShapeShapev3*
T0*
_output_shapes
:2&
$functional_1/dense_features/V3/Shape²
2functional_1/dense_features/V3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2functional_1/dense_features/V3/strided_slice/stack¶
4functional_1/dense_features/V3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_1/dense_features/V3/strided_slice/stack_1¶
4functional_1/dense_features/V3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_1/dense_features/V3/strided_slice/stack_2
,functional_1/dense_features/V3/strided_sliceStridedSlice-functional_1/dense_features/V3/Shape:output:0;functional_1/dense_features/V3/strided_slice/stack:output:0=functional_1/dense_features/V3/strided_slice/stack_1:output:0=functional_1/dense_features/V3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,functional_1/dense_features/V3/strided_slice¢
.functional_1/dense_features/V3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.functional_1/dense_features/V3/Reshape/shape/1
,functional_1/dense_features/V3/Reshape/shapePack5functional_1/dense_features/V3/strided_slice:output:07functional_1/dense_features/V3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,functional_1/dense_features/V3/Reshape/shapeÈ
&functional_1/dense_features/V3/ReshapeReshapev35functional_1/dense_features/V3/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&functional_1/dense_features/V3/Reshape~
$functional_1/dense_features/V4/ShapeShapev4*
T0*
_output_shapes
:2&
$functional_1/dense_features/V4/Shape²
2functional_1/dense_features/V4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2functional_1/dense_features/V4/strided_slice/stack¶
4functional_1/dense_features/V4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_1/dense_features/V4/strided_slice/stack_1¶
4functional_1/dense_features/V4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_1/dense_features/V4/strided_slice/stack_2
,functional_1/dense_features/V4/strided_sliceStridedSlice-functional_1/dense_features/V4/Shape:output:0;functional_1/dense_features/V4/strided_slice/stack:output:0=functional_1/dense_features/V4/strided_slice/stack_1:output:0=functional_1/dense_features/V4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,functional_1/dense_features/V4/strided_slice¢
.functional_1/dense_features/V4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.functional_1/dense_features/V4/Reshape/shape/1
,functional_1/dense_features/V4/Reshape/shapePack5functional_1/dense_features/V4/strided_slice:output:07functional_1/dense_features/V4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,functional_1/dense_features/V4/Reshape/shapeÈ
&functional_1/dense_features/V4/ReshapeReshapev45functional_1/dense_features/V4/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&functional_1/dense_features/V4/Reshape~
$functional_1/dense_features/V5/ShapeShapev5*
T0*
_output_shapes
:2&
$functional_1/dense_features/V5/Shape²
2functional_1/dense_features/V5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2functional_1/dense_features/V5/strided_slice/stack¶
4functional_1/dense_features/V5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_1/dense_features/V5/strided_slice/stack_1¶
4functional_1/dense_features/V5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_1/dense_features/V5/strided_slice/stack_2
,functional_1/dense_features/V5/strided_sliceStridedSlice-functional_1/dense_features/V5/Shape:output:0;functional_1/dense_features/V5/strided_slice/stack:output:0=functional_1/dense_features/V5/strided_slice/stack_1:output:0=functional_1/dense_features/V5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,functional_1/dense_features/V5/strided_slice¢
.functional_1/dense_features/V5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.functional_1/dense_features/V5/Reshape/shape/1
,functional_1/dense_features/V5/Reshape/shapePack5functional_1/dense_features/V5/strided_slice:output:07functional_1/dense_features/V5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,functional_1/dense_features/V5/Reshape/shapeÈ
&functional_1/dense_features/V5/ReshapeReshapev55functional_1/dense_features/V5/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&functional_1/dense_features/V5/Reshape~
$functional_1/dense_features/V6/ShapeShapev6*
T0*
_output_shapes
:2&
$functional_1/dense_features/V6/Shape²
2functional_1/dense_features/V6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2functional_1/dense_features/V6/strided_slice/stack¶
4functional_1/dense_features/V6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_1/dense_features/V6/strided_slice/stack_1¶
4functional_1/dense_features/V6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_1/dense_features/V6/strided_slice/stack_2
,functional_1/dense_features/V6/strided_sliceStridedSlice-functional_1/dense_features/V6/Shape:output:0;functional_1/dense_features/V6/strided_slice/stack:output:0=functional_1/dense_features/V6/strided_slice/stack_1:output:0=functional_1/dense_features/V6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,functional_1/dense_features/V6/strided_slice¢
.functional_1/dense_features/V6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.functional_1/dense_features/V6/Reshape/shape/1
,functional_1/dense_features/V6/Reshape/shapePack5functional_1/dense_features/V6/strided_slice:output:07functional_1/dense_features/V6/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,functional_1/dense_features/V6/Reshape/shapeÈ
&functional_1/dense_features/V6/ReshapeReshapev65functional_1/dense_features/V6/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&functional_1/dense_features/V6/Reshape~
$functional_1/dense_features/V7/ShapeShapev7*
T0*
_output_shapes
:2&
$functional_1/dense_features/V7/Shape²
2functional_1/dense_features/V7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2functional_1/dense_features/V7/strided_slice/stack¶
4functional_1/dense_features/V7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_1/dense_features/V7/strided_slice/stack_1¶
4functional_1/dense_features/V7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_1/dense_features/V7/strided_slice/stack_2
,functional_1/dense_features/V7/strided_sliceStridedSlice-functional_1/dense_features/V7/Shape:output:0;functional_1/dense_features/V7/strided_slice/stack:output:0=functional_1/dense_features/V7/strided_slice/stack_1:output:0=functional_1/dense_features/V7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,functional_1/dense_features/V7/strided_slice¢
.functional_1/dense_features/V7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.functional_1/dense_features/V7/Reshape/shape/1
,functional_1/dense_features/V7/Reshape/shapePack5functional_1/dense_features/V7/strided_slice:output:07functional_1/dense_features/V7/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,functional_1/dense_features/V7/Reshape/shapeÈ
&functional_1/dense_features/V7/ReshapeReshapev75functional_1/dense_features/V7/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&functional_1/dense_features/V7/Reshape~
$functional_1/dense_features/V8/ShapeShapev8*
T0*
_output_shapes
:2&
$functional_1/dense_features/V8/Shape²
2functional_1/dense_features/V8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2functional_1/dense_features/V8/strided_slice/stack¶
4functional_1/dense_features/V8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_1/dense_features/V8/strided_slice/stack_1¶
4functional_1/dense_features/V8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_1/dense_features/V8/strided_slice/stack_2
,functional_1/dense_features/V8/strided_sliceStridedSlice-functional_1/dense_features/V8/Shape:output:0;functional_1/dense_features/V8/strided_slice/stack:output:0=functional_1/dense_features/V8/strided_slice/stack_1:output:0=functional_1/dense_features/V8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,functional_1/dense_features/V8/strided_slice¢
.functional_1/dense_features/V8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.functional_1/dense_features/V8/Reshape/shape/1
,functional_1/dense_features/V8/Reshape/shapePack5functional_1/dense_features/V8/strided_slice:output:07functional_1/dense_features/V8/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,functional_1/dense_features/V8/Reshape/shapeÈ
&functional_1/dense_features/V8/ReshapeReshapev85functional_1/dense_features/V8/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&functional_1/dense_features/V8/Reshape~
$functional_1/dense_features/V9/ShapeShapev9*
T0*
_output_shapes
:2&
$functional_1/dense_features/V9/Shape²
2functional_1/dense_features/V9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2functional_1/dense_features/V9/strided_slice/stack¶
4functional_1/dense_features/V9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_1/dense_features/V9/strided_slice/stack_1¶
4functional_1/dense_features/V9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_1/dense_features/V9/strided_slice/stack_2
,functional_1/dense_features/V9/strided_sliceStridedSlice-functional_1/dense_features/V9/Shape:output:0;functional_1/dense_features/V9/strided_slice/stack:output:0=functional_1/dense_features/V9/strided_slice/stack_1:output:0=functional_1/dense_features/V9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,functional_1/dense_features/V9/strided_slice¢
.functional_1/dense_features/V9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.functional_1/dense_features/V9/Reshape/shape/1
,functional_1/dense_features/V9/Reshape/shapePack5functional_1/dense_features/V9/strided_slice:output:07functional_1/dense_features/V9/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,functional_1/dense_features/V9/Reshape/shapeÈ
&functional_1/dense_features/V9/ReshapeReshapev95functional_1/dense_features/V9/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&functional_1/dense_features/V9/Reshape
'functional_1/dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'functional_1/dense_features/concat/axis
"functional_1/dense_features/concatConcatV23functional_1/dense_features/Amount/Reshape:output:01functional_1/dense_features/Time/Reshape:output:0/functional_1/dense_features/V1/Reshape:output:00functional_1/dense_features/V10/Reshape:output:00functional_1/dense_features/V11/Reshape:output:00functional_1/dense_features/V12/Reshape:output:00functional_1/dense_features/V13/Reshape:output:00functional_1/dense_features/V14/Reshape:output:00functional_1/dense_features/V15/Reshape:output:00functional_1/dense_features/V16/Reshape:output:00functional_1/dense_features/V17/Reshape:output:00functional_1/dense_features/V18/Reshape:output:00functional_1/dense_features/V19/Reshape:output:0/functional_1/dense_features/V2/Reshape:output:00functional_1/dense_features/V20/Reshape:output:00functional_1/dense_features/V21/Reshape:output:00functional_1/dense_features/V22/Reshape:output:00functional_1/dense_features/V23/Reshape:output:00functional_1/dense_features/V24/Reshape:output:00functional_1/dense_features/V25/Reshape:output:00functional_1/dense_features/V26/Reshape:output:00functional_1/dense_features/V27/Reshape:output:00functional_1/dense_features/V28/Reshape:output:0/functional_1/dense_features/V3/Reshape:output:0/functional_1/dense_features/V4/Reshape:output:0/functional_1/dense_features/V5/Reshape:output:0/functional_1/dense_features/V6/Reshape:output:0/functional_1/dense_features/V7/Reshape:output:0/functional_1/dense_features/V8/Reshape:output:0/functional_1/dense_features/V9/Reshape:output:00functional_1/dense_features/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"functional_1/dense_features/concatõ
9functional_1/batch_normalization/batchnorm/ReadVariableOpReadVariableOpBfunctional_1_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02;
9functional_1/batch_normalization/batchnorm/ReadVariableOp©
0functional_1/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:22
0functional_1/batch_normalization/batchnorm/add/y
.functional_1/batch_normalization/batchnorm/addAddV2Afunctional_1/batch_normalization/batchnorm/ReadVariableOp:value:09functional_1/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:20
.functional_1/batch_normalization/batchnorm/addÆ
0functional_1/batch_normalization/batchnorm/RsqrtRsqrt2functional_1/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:22
0functional_1/batch_normalization/batchnorm/Rsqrt
=functional_1/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpFfunctional_1_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02?
=functional_1/batch_normalization/batchnorm/mul/ReadVariableOp
.functional_1/batch_normalization/batchnorm/mulMul4functional_1/batch_normalization/batchnorm/Rsqrt:y:0Efunctional_1/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:20
.functional_1/batch_normalization/batchnorm/mulþ
0functional_1/batch_normalization/batchnorm/mul_1Mul+functional_1/dense_features/concat:output:02functional_1/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0functional_1/batch_normalization/batchnorm/mul_1û
;functional_1/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpDfunctional_1_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02=
;functional_1/batch_normalization/batchnorm/ReadVariableOp_1
0functional_1/batch_normalization/batchnorm/mul_2MulCfunctional_1/batch_normalization/batchnorm/ReadVariableOp_1:value:02functional_1/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:22
0functional_1/batch_normalization/batchnorm/mul_2û
;functional_1/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpDfunctional_1_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02=
;functional_1/batch_normalization/batchnorm/ReadVariableOp_2
.functional_1/batch_normalization/batchnorm/subSubCfunctional_1/batch_normalization/batchnorm/ReadVariableOp_2:value:04functional_1/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:20
.functional_1/batch_normalization/batchnorm/sub
0functional_1/batch_normalization/batchnorm/add_1AddV24functional_1/batch_normalization/batchnorm/mul_1:z:02functional_1/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0functional_1/batch_normalization/batchnorm/add_1Æ
(functional_1/dense/MatMul/ReadVariableOpReadVariableOp1functional_1_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(functional_1/dense/MatMul/ReadVariableOpÚ
functional_1/dense/MatMulMatMul4functional_1/batch_normalization/batchnorm/add_1:z:00functional_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/dense/MatMulÅ
)functional_1/dense/BiasAdd/ReadVariableOpReadVariableOp2functional_1_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)functional_1/dense/BiasAdd/ReadVariableOpÍ
functional_1/dense/BiasAddBiasAdd#functional_1/dense/MatMul:product:01functional_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/dense/BiasAdd
functional_1/dense/SoftmaxSoftmax#functional_1/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/dense/Softmaxx
IdentityIdentity$functional_1/dense/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*ç
_input_shapesÕ
Ò:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameAmount:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameTime:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV1:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV10:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV11:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV12:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV13:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV14:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV15:L	H
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV16:L
H
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV17:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV18:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV19:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV2:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV20:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV21:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV22:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV23:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV24:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV25:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV26:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV27:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV28:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV3:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV4:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV5:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV6:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV7:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV8:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV9
¾
Ú
G__inference_functional_1_layer_call_and_return_conditional_losses_83010
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
	inputs_v9-
)batch_normalization_assignmovingavg_82978/
+batch_normalization_assignmovingavg_1_82984=
9batch_normalization_batchnorm_mul_readvariableop_resource9
5batch_normalization_batchnorm_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity¢7batch_normalization/AssignMovingAvg/AssignSubVariableOp¢9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpw
dense_features/Amount/ShapeShapeinputs_amount*
T0*
_output_shapes
:2
dense_features/Amount/Shape 
)dense_features/Amount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)dense_features/Amount/strided_slice/stack¤
+dense_features/Amount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features/Amount/strided_slice/stack_1¤
+dense_features/Amount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features/Amount/strided_slice/stack_2æ
#dense_features/Amount/strided_sliceStridedSlice$dense_features/Amount/Shape:output:02dense_features/Amount/strided_slice/stack:output:04dense_features/Amount/strided_slice/stack_1:output:04dense_features/Amount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#dense_features/Amount/strided_slice
%dense_features/Amount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%dense_features/Amount/Reshape/shape/1Þ
#dense_features/Amount/Reshape/shapePack,dense_features/Amount/strided_slice:output:0.dense_features/Amount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2%
#dense_features/Amount/Reshape/shape¸
dense_features/Amount/ReshapeReshapeinputs_amount,dense_features/Amount/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/Amount/Reshapeq
dense_features/Time/ShapeShapeinputs_time*
T0*
_output_shapes
:2
dense_features/Time/Shape
'dense_features/Time/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'dense_features/Time/strided_slice/stack 
)dense_features/Time/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)dense_features/Time/strided_slice/stack_1 
)dense_features/Time/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)dense_features/Time/strided_slice/stack_2Ú
!dense_features/Time/strided_sliceStridedSlice"dense_features/Time/Shape:output:00dense_features/Time/strided_slice/stack:output:02dense_features/Time/strided_slice/stack_1:output:02dense_features/Time/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!dense_features/Time/strided_slice
#dense_features/Time/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2%
#dense_features/Time/Reshape/shape/1Ö
!dense_features/Time/Reshape/shapePack*dense_features/Time/strided_slice:output:0,dense_features/Time/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2#
!dense_features/Time/Reshape/shape°
dense_features/Time/ReshapeReshapeinputs_time*dense_features/Time/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/Time/Reshapek
dense_features/V1/ShapeShape	inputs_v1*
T0*
_output_shapes
:2
dense_features/V1/Shape
%dense_features/V1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%dense_features/V1/strided_slice/stack
'dense_features/V1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V1/strided_slice/stack_1
'dense_features/V1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V1/strided_slice/stack_2Î
dense_features/V1/strided_sliceStridedSlice dense_features/V1/Shape:output:0.dense_features/V1/strided_slice/stack:output:00dense_features/V1/strided_slice/stack_1:output:00dense_features/V1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
dense_features/V1/strided_slice
!dense_features/V1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!dense_features/V1/Reshape/shape/1Î
dense_features/V1/Reshape/shapePack(dense_features/V1/strided_slice:output:0*dense_features/V1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
dense_features/V1/Reshape/shape¨
dense_features/V1/ReshapeReshape	inputs_v1(dense_features/V1/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V1/Reshapen
dense_features/V10/ShapeShape
inputs_v10*
T0*
_output_shapes
:2
dense_features/V10/Shape
&dense_features/V10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V10/strided_slice/stack
(dense_features/V10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V10/strided_slice/stack_1
(dense_features/V10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V10/strided_slice/stack_2Ô
 dense_features/V10/strided_sliceStridedSlice!dense_features/V10/Shape:output:0/dense_features/V10/strided_slice/stack:output:01dense_features/V10/strided_slice/stack_1:output:01dense_features/V10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V10/strided_slice
"dense_features/V10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V10/Reshape/shape/1Ò
 dense_features/V10/Reshape/shapePack)dense_features/V10/strided_slice:output:0+dense_features/V10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V10/Reshape/shape¬
dense_features/V10/ReshapeReshape
inputs_v10)dense_features/V10/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V10/Reshapen
dense_features/V11/ShapeShape
inputs_v11*
T0*
_output_shapes
:2
dense_features/V11/Shape
&dense_features/V11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V11/strided_slice/stack
(dense_features/V11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V11/strided_slice/stack_1
(dense_features/V11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V11/strided_slice/stack_2Ô
 dense_features/V11/strided_sliceStridedSlice!dense_features/V11/Shape:output:0/dense_features/V11/strided_slice/stack:output:01dense_features/V11/strided_slice/stack_1:output:01dense_features/V11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V11/strided_slice
"dense_features/V11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V11/Reshape/shape/1Ò
 dense_features/V11/Reshape/shapePack)dense_features/V11/strided_slice:output:0+dense_features/V11/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V11/Reshape/shape¬
dense_features/V11/ReshapeReshape
inputs_v11)dense_features/V11/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V11/Reshapen
dense_features/V12/ShapeShape
inputs_v12*
T0*
_output_shapes
:2
dense_features/V12/Shape
&dense_features/V12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V12/strided_slice/stack
(dense_features/V12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V12/strided_slice/stack_1
(dense_features/V12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V12/strided_slice/stack_2Ô
 dense_features/V12/strided_sliceStridedSlice!dense_features/V12/Shape:output:0/dense_features/V12/strided_slice/stack:output:01dense_features/V12/strided_slice/stack_1:output:01dense_features/V12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V12/strided_slice
"dense_features/V12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V12/Reshape/shape/1Ò
 dense_features/V12/Reshape/shapePack)dense_features/V12/strided_slice:output:0+dense_features/V12/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V12/Reshape/shape¬
dense_features/V12/ReshapeReshape
inputs_v12)dense_features/V12/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V12/Reshapen
dense_features/V13/ShapeShape
inputs_v13*
T0*
_output_shapes
:2
dense_features/V13/Shape
&dense_features/V13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V13/strided_slice/stack
(dense_features/V13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V13/strided_slice/stack_1
(dense_features/V13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V13/strided_slice/stack_2Ô
 dense_features/V13/strided_sliceStridedSlice!dense_features/V13/Shape:output:0/dense_features/V13/strided_slice/stack:output:01dense_features/V13/strided_slice/stack_1:output:01dense_features/V13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V13/strided_slice
"dense_features/V13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V13/Reshape/shape/1Ò
 dense_features/V13/Reshape/shapePack)dense_features/V13/strided_slice:output:0+dense_features/V13/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V13/Reshape/shape¬
dense_features/V13/ReshapeReshape
inputs_v13)dense_features/V13/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V13/Reshapen
dense_features/V14/ShapeShape
inputs_v14*
T0*
_output_shapes
:2
dense_features/V14/Shape
&dense_features/V14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V14/strided_slice/stack
(dense_features/V14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V14/strided_slice/stack_1
(dense_features/V14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V14/strided_slice/stack_2Ô
 dense_features/V14/strided_sliceStridedSlice!dense_features/V14/Shape:output:0/dense_features/V14/strided_slice/stack:output:01dense_features/V14/strided_slice/stack_1:output:01dense_features/V14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V14/strided_slice
"dense_features/V14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V14/Reshape/shape/1Ò
 dense_features/V14/Reshape/shapePack)dense_features/V14/strided_slice:output:0+dense_features/V14/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V14/Reshape/shape¬
dense_features/V14/ReshapeReshape
inputs_v14)dense_features/V14/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V14/Reshapen
dense_features/V15/ShapeShape
inputs_v15*
T0*
_output_shapes
:2
dense_features/V15/Shape
&dense_features/V15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V15/strided_slice/stack
(dense_features/V15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V15/strided_slice/stack_1
(dense_features/V15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V15/strided_slice/stack_2Ô
 dense_features/V15/strided_sliceStridedSlice!dense_features/V15/Shape:output:0/dense_features/V15/strided_slice/stack:output:01dense_features/V15/strided_slice/stack_1:output:01dense_features/V15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V15/strided_slice
"dense_features/V15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V15/Reshape/shape/1Ò
 dense_features/V15/Reshape/shapePack)dense_features/V15/strided_slice:output:0+dense_features/V15/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V15/Reshape/shape¬
dense_features/V15/ReshapeReshape
inputs_v15)dense_features/V15/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V15/Reshapen
dense_features/V16/ShapeShape
inputs_v16*
T0*
_output_shapes
:2
dense_features/V16/Shape
&dense_features/V16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V16/strided_slice/stack
(dense_features/V16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V16/strided_slice/stack_1
(dense_features/V16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V16/strided_slice/stack_2Ô
 dense_features/V16/strided_sliceStridedSlice!dense_features/V16/Shape:output:0/dense_features/V16/strided_slice/stack:output:01dense_features/V16/strided_slice/stack_1:output:01dense_features/V16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V16/strided_slice
"dense_features/V16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V16/Reshape/shape/1Ò
 dense_features/V16/Reshape/shapePack)dense_features/V16/strided_slice:output:0+dense_features/V16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V16/Reshape/shape¬
dense_features/V16/ReshapeReshape
inputs_v16)dense_features/V16/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V16/Reshapen
dense_features/V17/ShapeShape
inputs_v17*
T0*
_output_shapes
:2
dense_features/V17/Shape
&dense_features/V17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V17/strided_slice/stack
(dense_features/V17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V17/strided_slice/stack_1
(dense_features/V17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V17/strided_slice/stack_2Ô
 dense_features/V17/strided_sliceStridedSlice!dense_features/V17/Shape:output:0/dense_features/V17/strided_slice/stack:output:01dense_features/V17/strided_slice/stack_1:output:01dense_features/V17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V17/strided_slice
"dense_features/V17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V17/Reshape/shape/1Ò
 dense_features/V17/Reshape/shapePack)dense_features/V17/strided_slice:output:0+dense_features/V17/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V17/Reshape/shape¬
dense_features/V17/ReshapeReshape
inputs_v17)dense_features/V17/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V17/Reshapen
dense_features/V18/ShapeShape
inputs_v18*
T0*
_output_shapes
:2
dense_features/V18/Shape
&dense_features/V18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V18/strided_slice/stack
(dense_features/V18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V18/strided_slice/stack_1
(dense_features/V18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V18/strided_slice/stack_2Ô
 dense_features/V18/strided_sliceStridedSlice!dense_features/V18/Shape:output:0/dense_features/V18/strided_slice/stack:output:01dense_features/V18/strided_slice/stack_1:output:01dense_features/V18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V18/strided_slice
"dense_features/V18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V18/Reshape/shape/1Ò
 dense_features/V18/Reshape/shapePack)dense_features/V18/strided_slice:output:0+dense_features/V18/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V18/Reshape/shape¬
dense_features/V18/ReshapeReshape
inputs_v18)dense_features/V18/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V18/Reshapen
dense_features/V19/ShapeShape
inputs_v19*
T0*
_output_shapes
:2
dense_features/V19/Shape
&dense_features/V19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V19/strided_slice/stack
(dense_features/V19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V19/strided_slice/stack_1
(dense_features/V19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V19/strided_slice/stack_2Ô
 dense_features/V19/strided_sliceStridedSlice!dense_features/V19/Shape:output:0/dense_features/V19/strided_slice/stack:output:01dense_features/V19/strided_slice/stack_1:output:01dense_features/V19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V19/strided_slice
"dense_features/V19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V19/Reshape/shape/1Ò
 dense_features/V19/Reshape/shapePack)dense_features/V19/strided_slice:output:0+dense_features/V19/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V19/Reshape/shape¬
dense_features/V19/ReshapeReshape
inputs_v19)dense_features/V19/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V19/Reshapek
dense_features/V2/ShapeShape	inputs_v2*
T0*
_output_shapes
:2
dense_features/V2/Shape
%dense_features/V2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%dense_features/V2/strided_slice/stack
'dense_features/V2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V2/strided_slice/stack_1
'dense_features/V2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V2/strided_slice/stack_2Î
dense_features/V2/strided_sliceStridedSlice dense_features/V2/Shape:output:0.dense_features/V2/strided_slice/stack:output:00dense_features/V2/strided_slice/stack_1:output:00dense_features/V2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
dense_features/V2/strided_slice
!dense_features/V2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!dense_features/V2/Reshape/shape/1Î
dense_features/V2/Reshape/shapePack(dense_features/V2/strided_slice:output:0*dense_features/V2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
dense_features/V2/Reshape/shape¨
dense_features/V2/ReshapeReshape	inputs_v2(dense_features/V2/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V2/Reshapen
dense_features/V20/ShapeShape
inputs_v20*
T0*
_output_shapes
:2
dense_features/V20/Shape
&dense_features/V20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V20/strided_slice/stack
(dense_features/V20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V20/strided_slice/stack_1
(dense_features/V20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V20/strided_slice/stack_2Ô
 dense_features/V20/strided_sliceStridedSlice!dense_features/V20/Shape:output:0/dense_features/V20/strided_slice/stack:output:01dense_features/V20/strided_slice/stack_1:output:01dense_features/V20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V20/strided_slice
"dense_features/V20/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V20/Reshape/shape/1Ò
 dense_features/V20/Reshape/shapePack)dense_features/V20/strided_slice:output:0+dense_features/V20/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V20/Reshape/shape¬
dense_features/V20/ReshapeReshape
inputs_v20)dense_features/V20/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V20/Reshapen
dense_features/V21/ShapeShape
inputs_v21*
T0*
_output_shapes
:2
dense_features/V21/Shape
&dense_features/V21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V21/strided_slice/stack
(dense_features/V21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V21/strided_slice/stack_1
(dense_features/V21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V21/strided_slice/stack_2Ô
 dense_features/V21/strided_sliceStridedSlice!dense_features/V21/Shape:output:0/dense_features/V21/strided_slice/stack:output:01dense_features/V21/strided_slice/stack_1:output:01dense_features/V21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V21/strided_slice
"dense_features/V21/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V21/Reshape/shape/1Ò
 dense_features/V21/Reshape/shapePack)dense_features/V21/strided_slice:output:0+dense_features/V21/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V21/Reshape/shape¬
dense_features/V21/ReshapeReshape
inputs_v21)dense_features/V21/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V21/Reshapen
dense_features/V22/ShapeShape
inputs_v22*
T0*
_output_shapes
:2
dense_features/V22/Shape
&dense_features/V22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V22/strided_slice/stack
(dense_features/V22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V22/strided_slice/stack_1
(dense_features/V22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V22/strided_slice/stack_2Ô
 dense_features/V22/strided_sliceStridedSlice!dense_features/V22/Shape:output:0/dense_features/V22/strided_slice/stack:output:01dense_features/V22/strided_slice/stack_1:output:01dense_features/V22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V22/strided_slice
"dense_features/V22/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V22/Reshape/shape/1Ò
 dense_features/V22/Reshape/shapePack)dense_features/V22/strided_slice:output:0+dense_features/V22/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V22/Reshape/shape¬
dense_features/V22/ReshapeReshape
inputs_v22)dense_features/V22/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V22/Reshapen
dense_features/V23/ShapeShape
inputs_v23*
T0*
_output_shapes
:2
dense_features/V23/Shape
&dense_features/V23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V23/strided_slice/stack
(dense_features/V23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V23/strided_slice/stack_1
(dense_features/V23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V23/strided_slice/stack_2Ô
 dense_features/V23/strided_sliceStridedSlice!dense_features/V23/Shape:output:0/dense_features/V23/strided_slice/stack:output:01dense_features/V23/strided_slice/stack_1:output:01dense_features/V23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V23/strided_slice
"dense_features/V23/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V23/Reshape/shape/1Ò
 dense_features/V23/Reshape/shapePack)dense_features/V23/strided_slice:output:0+dense_features/V23/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V23/Reshape/shape¬
dense_features/V23/ReshapeReshape
inputs_v23)dense_features/V23/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V23/Reshapen
dense_features/V24/ShapeShape
inputs_v24*
T0*
_output_shapes
:2
dense_features/V24/Shape
&dense_features/V24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V24/strided_slice/stack
(dense_features/V24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V24/strided_slice/stack_1
(dense_features/V24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V24/strided_slice/stack_2Ô
 dense_features/V24/strided_sliceStridedSlice!dense_features/V24/Shape:output:0/dense_features/V24/strided_slice/stack:output:01dense_features/V24/strided_slice/stack_1:output:01dense_features/V24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V24/strided_slice
"dense_features/V24/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V24/Reshape/shape/1Ò
 dense_features/V24/Reshape/shapePack)dense_features/V24/strided_slice:output:0+dense_features/V24/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V24/Reshape/shape¬
dense_features/V24/ReshapeReshape
inputs_v24)dense_features/V24/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V24/Reshapen
dense_features/V25/ShapeShape
inputs_v25*
T0*
_output_shapes
:2
dense_features/V25/Shape
&dense_features/V25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V25/strided_slice/stack
(dense_features/V25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V25/strided_slice/stack_1
(dense_features/V25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V25/strided_slice/stack_2Ô
 dense_features/V25/strided_sliceStridedSlice!dense_features/V25/Shape:output:0/dense_features/V25/strided_slice/stack:output:01dense_features/V25/strided_slice/stack_1:output:01dense_features/V25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V25/strided_slice
"dense_features/V25/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V25/Reshape/shape/1Ò
 dense_features/V25/Reshape/shapePack)dense_features/V25/strided_slice:output:0+dense_features/V25/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V25/Reshape/shape¬
dense_features/V25/ReshapeReshape
inputs_v25)dense_features/V25/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V25/Reshapen
dense_features/V26/ShapeShape
inputs_v26*
T0*
_output_shapes
:2
dense_features/V26/Shape
&dense_features/V26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V26/strided_slice/stack
(dense_features/V26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V26/strided_slice/stack_1
(dense_features/V26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V26/strided_slice/stack_2Ô
 dense_features/V26/strided_sliceStridedSlice!dense_features/V26/Shape:output:0/dense_features/V26/strided_slice/stack:output:01dense_features/V26/strided_slice/stack_1:output:01dense_features/V26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V26/strided_slice
"dense_features/V26/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V26/Reshape/shape/1Ò
 dense_features/V26/Reshape/shapePack)dense_features/V26/strided_slice:output:0+dense_features/V26/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V26/Reshape/shape¬
dense_features/V26/ReshapeReshape
inputs_v26)dense_features/V26/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V26/Reshapen
dense_features/V27/ShapeShape
inputs_v27*
T0*
_output_shapes
:2
dense_features/V27/Shape
&dense_features/V27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V27/strided_slice/stack
(dense_features/V27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V27/strided_slice/stack_1
(dense_features/V27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V27/strided_slice/stack_2Ô
 dense_features/V27/strided_sliceStridedSlice!dense_features/V27/Shape:output:0/dense_features/V27/strided_slice/stack:output:01dense_features/V27/strided_slice/stack_1:output:01dense_features/V27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V27/strided_slice
"dense_features/V27/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V27/Reshape/shape/1Ò
 dense_features/V27/Reshape/shapePack)dense_features/V27/strided_slice:output:0+dense_features/V27/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V27/Reshape/shape¬
dense_features/V27/ReshapeReshape
inputs_v27)dense_features/V27/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V27/Reshapen
dense_features/V28/ShapeShape
inputs_v28*
T0*
_output_shapes
:2
dense_features/V28/Shape
&dense_features/V28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V28/strided_slice/stack
(dense_features/V28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V28/strided_slice/stack_1
(dense_features/V28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V28/strided_slice/stack_2Ô
 dense_features/V28/strided_sliceStridedSlice!dense_features/V28/Shape:output:0/dense_features/V28/strided_slice/stack:output:01dense_features/V28/strided_slice/stack_1:output:01dense_features/V28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V28/strided_slice
"dense_features/V28/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V28/Reshape/shape/1Ò
 dense_features/V28/Reshape/shapePack)dense_features/V28/strided_slice:output:0+dense_features/V28/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V28/Reshape/shape¬
dense_features/V28/ReshapeReshape
inputs_v28)dense_features/V28/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V28/Reshapek
dense_features/V3/ShapeShape	inputs_v3*
T0*
_output_shapes
:2
dense_features/V3/Shape
%dense_features/V3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%dense_features/V3/strided_slice/stack
'dense_features/V3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V3/strided_slice/stack_1
'dense_features/V3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V3/strided_slice/stack_2Î
dense_features/V3/strided_sliceStridedSlice dense_features/V3/Shape:output:0.dense_features/V3/strided_slice/stack:output:00dense_features/V3/strided_slice/stack_1:output:00dense_features/V3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
dense_features/V3/strided_slice
!dense_features/V3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!dense_features/V3/Reshape/shape/1Î
dense_features/V3/Reshape/shapePack(dense_features/V3/strided_slice:output:0*dense_features/V3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
dense_features/V3/Reshape/shape¨
dense_features/V3/ReshapeReshape	inputs_v3(dense_features/V3/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V3/Reshapek
dense_features/V4/ShapeShape	inputs_v4*
T0*
_output_shapes
:2
dense_features/V4/Shape
%dense_features/V4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%dense_features/V4/strided_slice/stack
'dense_features/V4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V4/strided_slice/stack_1
'dense_features/V4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V4/strided_slice/stack_2Î
dense_features/V4/strided_sliceStridedSlice dense_features/V4/Shape:output:0.dense_features/V4/strided_slice/stack:output:00dense_features/V4/strided_slice/stack_1:output:00dense_features/V4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
dense_features/V4/strided_slice
!dense_features/V4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!dense_features/V4/Reshape/shape/1Î
dense_features/V4/Reshape/shapePack(dense_features/V4/strided_slice:output:0*dense_features/V4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
dense_features/V4/Reshape/shape¨
dense_features/V4/ReshapeReshape	inputs_v4(dense_features/V4/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V4/Reshapek
dense_features/V5/ShapeShape	inputs_v5*
T0*
_output_shapes
:2
dense_features/V5/Shape
%dense_features/V5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%dense_features/V5/strided_slice/stack
'dense_features/V5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V5/strided_slice/stack_1
'dense_features/V5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V5/strided_slice/stack_2Î
dense_features/V5/strided_sliceStridedSlice dense_features/V5/Shape:output:0.dense_features/V5/strided_slice/stack:output:00dense_features/V5/strided_slice/stack_1:output:00dense_features/V5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
dense_features/V5/strided_slice
!dense_features/V5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!dense_features/V5/Reshape/shape/1Î
dense_features/V5/Reshape/shapePack(dense_features/V5/strided_slice:output:0*dense_features/V5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
dense_features/V5/Reshape/shape¨
dense_features/V5/ReshapeReshape	inputs_v5(dense_features/V5/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V5/Reshapek
dense_features/V6/ShapeShape	inputs_v6*
T0*
_output_shapes
:2
dense_features/V6/Shape
%dense_features/V6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%dense_features/V6/strided_slice/stack
'dense_features/V6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V6/strided_slice/stack_1
'dense_features/V6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V6/strided_slice/stack_2Î
dense_features/V6/strided_sliceStridedSlice dense_features/V6/Shape:output:0.dense_features/V6/strided_slice/stack:output:00dense_features/V6/strided_slice/stack_1:output:00dense_features/V6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
dense_features/V6/strided_slice
!dense_features/V6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!dense_features/V6/Reshape/shape/1Î
dense_features/V6/Reshape/shapePack(dense_features/V6/strided_slice:output:0*dense_features/V6/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
dense_features/V6/Reshape/shape¨
dense_features/V6/ReshapeReshape	inputs_v6(dense_features/V6/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V6/Reshapek
dense_features/V7/ShapeShape	inputs_v7*
T0*
_output_shapes
:2
dense_features/V7/Shape
%dense_features/V7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%dense_features/V7/strided_slice/stack
'dense_features/V7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V7/strided_slice/stack_1
'dense_features/V7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V7/strided_slice/stack_2Î
dense_features/V7/strided_sliceStridedSlice dense_features/V7/Shape:output:0.dense_features/V7/strided_slice/stack:output:00dense_features/V7/strided_slice/stack_1:output:00dense_features/V7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
dense_features/V7/strided_slice
!dense_features/V7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!dense_features/V7/Reshape/shape/1Î
dense_features/V7/Reshape/shapePack(dense_features/V7/strided_slice:output:0*dense_features/V7/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
dense_features/V7/Reshape/shape¨
dense_features/V7/ReshapeReshape	inputs_v7(dense_features/V7/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V7/Reshapek
dense_features/V8/ShapeShape	inputs_v8*
T0*
_output_shapes
:2
dense_features/V8/Shape
%dense_features/V8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%dense_features/V8/strided_slice/stack
'dense_features/V8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V8/strided_slice/stack_1
'dense_features/V8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V8/strided_slice/stack_2Î
dense_features/V8/strided_sliceStridedSlice dense_features/V8/Shape:output:0.dense_features/V8/strided_slice/stack:output:00dense_features/V8/strided_slice/stack_1:output:00dense_features/V8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
dense_features/V8/strided_slice
!dense_features/V8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!dense_features/V8/Reshape/shape/1Î
dense_features/V8/Reshape/shapePack(dense_features/V8/strided_slice:output:0*dense_features/V8/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
dense_features/V8/Reshape/shape¨
dense_features/V8/ReshapeReshape	inputs_v8(dense_features/V8/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V8/Reshapek
dense_features/V9/ShapeShape	inputs_v9*
T0*
_output_shapes
:2
dense_features/V9/Shape
%dense_features/V9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%dense_features/V9/strided_slice/stack
'dense_features/V9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V9/strided_slice/stack_1
'dense_features/V9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V9/strided_slice/stack_2Î
dense_features/V9/strided_sliceStridedSlice dense_features/V9/Shape:output:0.dense_features/V9/strided_slice/stack:output:00dense_features/V9/strided_slice/stack_1:output:00dense_features/V9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
dense_features/V9/strided_slice
!dense_features/V9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!dense_features/V9/Reshape/shape/1Î
dense_features/V9/Reshape/shapePack(dense_features/V9/strided_slice:output:0*dense_features/V9/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
dense_features/V9/Reshape/shape¨
dense_features/V9/ReshapeReshape	inputs_v9(dense_features/V9/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V9/Reshape
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
dense_features/concat/axisë	
dense_features/concatConcatV2&dense_features/Amount/Reshape:output:0$dense_features/Time/Reshape:output:0"dense_features/V1/Reshape:output:0#dense_features/V10/Reshape:output:0#dense_features/V11/Reshape:output:0#dense_features/V12/Reshape:output:0#dense_features/V13/Reshape:output:0#dense_features/V14/Reshape:output:0#dense_features/V15/Reshape:output:0#dense_features/V16/Reshape:output:0#dense_features/V17/Reshape:output:0#dense_features/V18/Reshape:output:0#dense_features/V19/Reshape:output:0"dense_features/V2/Reshape:output:0#dense_features/V20/Reshape:output:0#dense_features/V21/Reshape:output:0#dense_features/V22/Reshape:output:0#dense_features/V23/Reshape:output:0#dense_features/V24/Reshape:output:0#dense_features/V25/Reshape:output:0#dense_features/V26/Reshape:output:0#dense_features/V27/Reshape:output:0#dense_features/V28/Reshape:output:0"dense_features/V3/Reshape:output:0"dense_features/V4/Reshape:output:0"dense_features/V5/Reshape:output:0"dense_features/V6/Reshape:output:0"dense_features/V7/Reshape:output:0"dense_features/V8/Reshape:output:0"dense_features/V9/Reshape:output:0#dense_features/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/concat²
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 24
2batch_normalization/moments/mean/reduction_indicesã
 batch_normalization/moments/meanMeandense_features/concat:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2"
 batch_normalization/moments/mean¸
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes

:2*
(batch_normalization/moments/StopGradientø
-batch_normalization/moments/SquaredDifferenceSquaredDifferencedense_features/concat:output:01batch_normalization/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-batch_normalization/moments/SquaredDifferenceº
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization/moments/variance/reduction_indices
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2&
$batch_normalization/moments/variance¼
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2%
#batch_normalization/moments/SqueezeÄ
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1Ù
)batch_normalization/AssignMovingAvg/decayConst*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/82978*
_output_shapes
: *
dtype0*
valueB
 *
×#<2+
)batch_normalization/AssignMovingAvg/decayÎ
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp)batch_normalization_assignmovingavg_82978*
_output_shapes
:*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp¦
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/82978*
_output_shapes
:2)
'batch_normalization/AssignMovingAvg/sub
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/82978*
_output_shapes
:2)
'batch_normalization/AssignMovingAvg/mul÷
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp)batch_normalization_assignmovingavg_82978+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/82978*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOpß
+batch_normalization/AssignMovingAvg_1/decayConst*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/82984*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization/AssignMovingAvg_1/decayÔ
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp+batch_normalization_assignmovingavg_1_82984*
_output_shapes
:*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp°
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/82984*
_output_shapes
:2+
)batch_normalization/AssignMovingAvg_1/sub§
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/82984*
_output_shapes
:2+
)batch_normalization/AssignMovingAvg_1/mul
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp+batch_normalization_assignmovingavg_1_82984-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/82984*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2%
#batch_normalization/batchnorm/add/yÒ
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/add
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/RsqrtÚ
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOpÕ
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/mulÊ
#batch_normalization/batchnorm/mul_1Muldense_features/concat:output:0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#batch_normalization/batchnorm/mul_1Ë
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/mul_2Î
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization/batchnorm/ReadVariableOpÑ
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/subÕ
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#batch_normalization/batchnorm/add_1
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp¦
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAdds
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Softmaxá
IdentityIdentitydense/Softmax:softmax:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*ç
_input_shapesÕ
Ò:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameinputs/Amount:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinputs/Time:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V1:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V10:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V11:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V12:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V13:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V14:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V15:S	O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V16:S
O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V17:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V18:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V19:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V2:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V20:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V21:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V22:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V23:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V24:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V25:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V26:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V27:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V28:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V3:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V4:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V5:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V6:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V7:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V8:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V9
þ 
º
,__inference_functional_1_layer_call_fn_82642

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
v9
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallamounttimev1v10v11v12v13v14v15v16v17v18v19v2v20v21v22v23v24v25v26v27v28v3v4v5v6v7v8v9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

 !"#*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_826272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*ç
_input_shapesÕ
Ò:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameAmount:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameTime:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV1:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV10:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV11:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV12:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV13:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV14:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV15:L	H
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV16:L
H
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV17:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV18:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV19:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV2:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV20:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV21:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV22:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV23:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV24:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV25:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV26:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV27:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV28:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV3:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV4:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV5:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV6:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV7:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV8:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV9
Ô
z
%__inference_dense_layer_call_fn_84120

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_823882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
¨
@__inference_dense_layer_call_and_return_conditional_losses_82388

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â$
Õ
.__inference_dense_features_layer_call_fn_83984
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
identityû
PartitionedCallPartitionedCallfeatures_amountfeatures_timefeatures_v1features_v10features_v11features_v12features_v13features_v14features_v15features_v16features_v17features_v18features_v19features_v2features_v20features_v21features_v22features_v23features_v24features_v25features_v26features_v27features_v28features_v3features_v4features_v5features_v6features_v7features_v8features_v9*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_819962
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ï
_input_shapes½
º:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:X T
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_namefeatures/Amount:VR
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namefeatures/Time:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V1:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V10:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V11:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V12:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V13:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V14:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V15:U	Q
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V16:U
Q
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V17:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V18:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V19:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V2:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V20:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V21:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V22:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V23:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V24:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V25:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V26:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V27:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V28:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V3:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V4:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V5:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V6:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V7:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V8:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V9
ä
þ
G__inference_functional_1_layer_call_and_return_conditional_losses_83308
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
	inputs_v99
5batch_normalization_batchnorm_readvariableop_resource=
9batch_normalization_batchnorm_mul_readvariableop_resource;
7batch_normalization_batchnorm_readvariableop_1_resource;
7batch_normalization_batchnorm_readvariableop_2_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityw
dense_features/Amount/ShapeShapeinputs_amount*
T0*
_output_shapes
:2
dense_features/Amount/Shape 
)dense_features/Amount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)dense_features/Amount/strided_slice/stack¤
+dense_features/Amount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features/Amount/strided_slice/stack_1¤
+dense_features/Amount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features/Amount/strided_slice/stack_2æ
#dense_features/Amount/strided_sliceStridedSlice$dense_features/Amount/Shape:output:02dense_features/Amount/strided_slice/stack:output:04dense_features/Amount/strided_slice/stack_1:output:04dense_features/Amount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#dense_features/Amount/strided_slice
%dense_features/Amount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%dense_features/Amount/Reshape/shape/1Þ
#dense_features/Amount/Reshape/shapePack,dense_features/Amount/strided_slice:output:0.dense_features/Amount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2%
#dense_features/Amount/Reshape/shape¸
dense_features/Amount/ReshapeReshapeinputs_amount,dense_features/Amount/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/Amount/Reshapeq
dense_features/Time/ShapeShapeinputs_time*
T0*
_output_shapes
:2
dense_features/Time/Shape
'dense_features/Time/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'dense_features/Time/strided_slice/stack 
)dense_features/Time/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)dense_features/Time/strided_slice/stack_1 
)dense_features/Time/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)dense_features/Time/strided_slice/stack_2Ú
!dense_features/Time/strided_sliceStridedSlice"dense_features/Time/Shape:output:00dense_features/Time/strided_slice/stack:output:02dense_features/Time/strided_slice/stack_1:output:02dense_features/Time/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!dense_features/Time/strided_slice
#dense_features/Time/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2%
#dense_features/Time/Reshape/shape/1Ö
!dense_features/Time/Reshape/shapePack*dense_features/Time/strided_slice:output:0,dense_features/Time/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2#
!dense_features/Time/Reshape/shape°
dense_features/Time/ReshapeReshapeinputs_time*dense_features/Time/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/Time/Reshapek
dense_features/V1/ShapeShape	inputs_v1*
T0*
_output_shapes
:2
dense_features/V1/Shape
%dense_features/V1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%dense_features/V1/strided_slice/stack
'dense_features/V1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V1/strided_slice/stack_1
'dense_features/V1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V1/strided_slice/stack_2Î
dense_features/V1/strided_sliceStridedSlice dense_features/V1/Shape:output:0.dense_features/V1/strided_slice/stack:output:00dense_features/V1/strided_slice/stack_1:output:00dense_features/V1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
dense_features/V1/strided_slice
!dense_features/V1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!dense_features/V1/Reshape/shape/1Î
dense_features/V1/Reshape/shapePack(dense_features/V1/strided_slice:output:0*dense_features/V1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
dense_features/V1/Reshape/shape¨
dense_features/V1/ReshapeReshape	inputs_v1(dense_features/V1/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V1/Reshapen
dense_features/V10/ShapeShape
inputs_v10*
T0*
_output_shapes
:2
dense_features/V10/Shape
&dense_features/V10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V10/strided_slice/stack
(dense_features/V10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V10/strided_slice/stack_1
(dense_features/V10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V10/strided_slice/stack_2Ô
 dense_features/V10/strided_sliceStridedSlice!dense_features/V10/Shape:output:0/dense_features/V10/strided_slice/stack:output:01dense_features/V10/strided_slice/stack_1:output:01dense_features/V10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V10/strided_slice
"dense_features/V10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V10/Reshape/shape/1Ò
 dense_features/V10/Reshape/shapePack)dense_features/V10/strided_slice:output:0+dense_features/V10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V10/Reshape/shape¬
dense_features/V10/ReshapeReshape
inputs_v10)dense_features/V10/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V10/Reshapen
dense_features/V11/ShapeShape
inputs_v11*
T0*
_output_shapes
:2
dense_features/V11/Shape
&dense_features/V11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V11/strided_slice/stack
(dense_features/V11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V11/strided_slice/stack_1
(dense_features/V11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V11/strided_slice/stack_2Ô
 dense_features/V11/strided_sliceStridedSlice!dense_features/V11/Shape:output:0/dense_features/V11/strided_slice/stack:output:01dense_features/V11/strided_slice/stack_1:output:01dense_features/V11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V11/strided_slice
"dense_features/V11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V11/Reshape/shape/1Ò
 dense_features/V11/Reshape/shapePack)dense_features/V11/strided_slice:output:0+dense_features/V11/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V11/Reshape/shape¬
dense_features/V11/ReshapeReshape
inputs_v11)dense_features/V11/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V11/Reshapen
dense_features/V12/ShapeShape
inputs_v12*
T0*
_output_shapes
:2
dense_features/V12/Shape
&dense_features/V12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V12/strided_slice/stack
(dense_features/V12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V12/strided_slice/stack_1
(dense_features/V12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V12/strided_slice/stack_2Ô
 dense_features/V12/strided_sliceStridedSlice!dense_features/V12/Shape:output:0/dense_features/V12/strided_slice/stack:output:01dense_features/V12/strided_slice/stack_1:output:01dense_features/V12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V12/strided_slice
"dense_features/V12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V12/Reshape/shape/1Ò
 dense_features/V12/Reshape/shapePack)dense_features/V12/strided_slice:output:0+dense_features/V12/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V12/Reshape/shape¬
dense_features/V12/ReshapeReshape
inputs_v12)dense_features/V12/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V12/Reshapen
dense_features/V13/ShapeShape
inputs_v13*
T0*
_output_shapes
:2
dense_features/V13/Shape
&dense_features/V13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V13/strided_slice/stack
(dense_features/V13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V13/strided_slice/stack_1
(dense_features/V13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V13/strided_slice/stack_2Ô
 dense_features/V13/strided_sliceStridedSlice!dense_features/V13/Shape:output:0/dense_features/V13/strided_slice/stack:output:01dense_features/V13/strided_slice/stack_1:output:01dense_features/V13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V13/strided_slice
"dense_features/V13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V13/Reshape/shape/1Ò
 dense_features/V13/Reshape/shapePack)dense_features/V13/strided_slice:output:0+dense_features/V13/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V13/Reshape/shape¬
dense_features/V13/ReshapeReshape
inputs_v13)dense_features/V13/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V13/Reshapen
dense_features/V14/ShapeShape
inputs_v14*
T0*
_output_shapes
:2
dense_features/V14/Shape
&dense_features/V14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V14/strided_slice/stack
(dense_features/V14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V14/strided_slice/stack_1
(dense_features/V14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V14/strided_slice/stack_2Ô
 dense_features/V14/strided_sliceStridedSlice!dense_features/V14/Shape:output:0/dense_features/V14/strided_slice/stack:output:01dense_features/V14/strided_slice/stack_1:output:01dense_features/V14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V14/strided_slice
"dense_features/V14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V14/Reshape/shape/1Ò
 dense_features/V14/Reshape/shapePack)dense_features/V14/strided_slice:output:0+dense_features/V14/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V14/Reshape/shape¬
dense_features/V14/ReshapeReshape
inputs_v14)dense_features/V14/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V14/Reshapen
dense_features/V15/ShapeShape
inputs_v15*
T0*
_output_shapes
:2
dense_features/V15/Shape
&dense_features/V15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V15/strided_slice/stack
(dense_features/V15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V15/strided_slice/stack_1
(dense_features/V15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V15/strided_slice/stack_2Ô
 dense_features/V15/strided_sliceStridedSlice!dense_features/V15/Shape:output:0/dense_features/V15/strided_slice/stack:output:01dense_features/V15/strided_slice/stack_1:output:01dense_features/V15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V15/strided_slice
"dense_features/V15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V15/Reshape/shape/1Ò
 dense_features/V15/Reshape/shapePack)dense_features/V15/strided_slice:output:0+dense_features/V15/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V15/Reshape/shape¬
dense_features/V15/ReshapeReshape
inputs_v15)dense_features/V15/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V15/Reshapen
dense_features/V16/ShapeShape
inputs_v16*
T0*
_output_shapes
:2
dense_features/V16/Shape
&dense_features/V16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V16/strided_slice/stack
(dense_features/V16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V16/strided_slice/stack_1
(dense_features/V16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V16/strided_slice/stack_2Ô
 dense_features/V16/strided_sliceStridedSlice!dense_features/V16/Shape:output:0/dense_features/V16/strided_slice/stack:output:01dense_features/V16/strided_slice/stack_1:output:01dense_features/V16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V16/strided_slice
"dense_features/V16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V16/Reshape/shape/1Ò
 dense_features/V16/Reshape/shapePack)dense_features/V16/strided_slice:output:0+dense_features/V16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V16/Reshape/shape¬
dense_features/V16/ReshapeReshape
inputs_v16)dense_features/V16/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V16/Reshapen
dense_features/V17/ShapeShape
inputs_v17*
T0*
_output_shapes
:2
dense_features/V17/Shape
&dense_features/V17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V17/strided_slice/stack
(dense_features/V17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V17/strided_slice/stack_1
(dense_features/V17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V17/strided_slice/stack_2Ô
 dense_features/V17/strided_sliceStridedSlice!dense_features/V17/Shape:output:0/dense_features/V17/strided_slice/stack:output:01dense_features/V17/strided_slice/stack_1:output:01dense_features/V17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V17/strided_slice
"dense_features/V17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V17/Reshape/shape/1Ò
 dense_features/V17/Reshape/shapePack)dense_features/V17/strided_slice:output:0+dense_features/V17/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V17/Reshape/shape¬
dense_features/V17/ReshapeReshape
inputs_v17)dense_features/V17/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V17/Reshapen
dense_features/V18/ShapeShape
inputs_v18*
T0*
_output_shapes
:2
dense_features/V18/Shape
&dense_features/V18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V18/strided_slice/stack
(dense_features/V18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V18/strided_slice/stack_1
(dense_features/V18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V18/strided_slice/stack_2Ô
 dense_features/V18/strided_sliceStridedSlice!dense_features/V18/Shape:output:0/dense_features/V18/strided_slice/stack:output:01dense_features/V18/strided_slice/stack_1:output:01dense_features/V18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V18/strided_slice
"dense_features/V18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V18/Reshape/shape/1Ò
 dense_features/V18/Reshape/shapePack)dense_features/V18/strided_slice:output:0+dense_features/V18/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V18/Reshape/shape¬
dense_features/V18/ReshapeReshape
inputs_v18)dense_features/V18/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V18/Reshapen
dense_features/V19/ShapeShape
inputs_v19*
T0*
_output_shapes
:2
dense_features/V19/Shape
&dense_features/V19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V19/strided_slice/stack
(dense_features/V19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V19/strided_slice/stack_1
(dense_features/V19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V19/strided_slice/stack_2Ô
 dense_features/V19/strided_sliceStridedSlice!dense_features/V19/Shape:output:0/dense_features/V19/strided_slice/stack:output:01dense_features/V19/strided_slice/stack_1:output:01dense_features/V19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V19/strided_slice
"dense_features/V19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V19/Reshape/shape/1Ò
 dense_features/V19/Reshape/shapePack)dense_features/V19/strided_slice:output:0+dense_features/V19/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V19/Reshape/shape¬
dense_features/V19/ReshapeReshape
inputs_v19)dense_features/V19/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V19/Reshapek
dense_features/V2/ShapeShape	inputs_v2*
T0*
_output_shapes
:2
dense_features/V2/Shape
%dense_features/V2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%dense_features/V2/strided_slice/stack
'dense_features/V2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V2/strided_slice/stack_1
'dense_features/V2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V2/strided_slice/stack_2Î
dense_features/V2/strided_sliceStridedSlice dense_features/V2/Shape:output:0.dense_features/V2/strided_slice/stack:output:00dense_features/V2/strided_slice/stack_1:output:00dense_features/V2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
dense_features/V2/strided_slice
!dense_features/V2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!dense_features/V2/Reshape/shape/1Î
dense_features/V2/Reshape/shapePack(dense_features/V2/strided_slice:output:0*dense_features/V2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
dense_features/V2/Reshape/shape¨
dense_features/V2/ReshapeReshape	inputs_v2(dense_features/V2/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V2/Reshapen
dense_features/V20/ShapeShape
inputs_v20*
T0*
_output_shapes
:2
dense_features/V20/Shape
&dense_features/V20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V20/strided_slice/stack
(dense_features/V20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V20/strided_slice/stack_1
(dense_features/V20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V20/strided_slice/stack_2Ô
 dense_features/V20/strided_sliceStridedSlice!dense_features/V20/Shape:output:0/dense_features/V20/strided_slice/stack:output:01dense_features/V20/strided_slice/stack_1:output:01dense_features/V20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V20/strided_slice
"dense_features/V20/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V20/Reshape/shape/1Ò
 dense_features/V20/Reshape/shapePack)dense_features/V20/strided_slice:output:0+dense_features/V20/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V20/Reshape/shape¬
dense_features/V20/ReshapeReshape
inputs_v20)dense_features/V20/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V20/Reshapen
dense_features/V21/ShapeShape
inputs_v21*
T0*
_output_shapes
:2
dense_features/V21/Shape
&dense_features/V21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V21/strided_slice/stack
(dense_features/V21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V21/strided_slice/stack_1
(dense_features/V21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V21/strided_slice/stack_2Ô
 dense_features/V21/strided_sliceStridedSlice!dense_features/V21/Shape:output:0/dense_features/V21/strided_slice/stack:output:01dense_features/V21/strided_slice/stack_1:output:01dense_features/V21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V21/strided_slice
"dense_features/V21/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V21/Reshape/shape/1Ò
 dense_features/V21/Reshape/shapePack)dense_features/V21/strided_slice:output:0+dense_features/V21/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V21/Reshape/shape¬
dense_features/V21/ReshapeReshape
inputs_v21)dense_features/V21/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V21/Reshapen
dense_features/V22/ShapeShape
inputs_v22*
T0*
_output_shapes
:2
dense_features/V22/Shape
&dense_features/V22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V22/strided_slice/stack
(dense_features/V22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V22/strided_slice/stack_1
(dense_features/V22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V22/strided_slice/stack_2Ô
 dense_features/V22/strided_sliceStridedSlice!dense_features/V22/Shape:output:0/dense_features/V22/strided_slice/stack:output:01dense_features/V22/strided_slice/stack_1:output:01dense_features/V22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V22/strided_slice
"dense_features/V22/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V22/Reshape/shape/1Ò
 dense_features/V22/Reshape/shapePack)dense_features/V22/strided_slice:output:0+dense_features/V22/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V22/Reshape/shape¬
dense_features/V22/ReshapeReshape
inputs_v22)dense_features/V22/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V22/Reshapen
dense_features/V23/ShapeShape
inputs_v23*
T0*
_output_shapes
:2
dense_features/V23/Shape
&dense_features/V23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V23/strided_slice/stack
(dense_features/V23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V23/strided_slice/stack_1
(dense_features/V23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V23/strided_slice/stack_2Ô
 dense_features/V23/strided_sliceStridedSlice!dense_features/V23/Shape:output:0/dense_features/V23/strided_slice/stack:output:01dense_features/V23/strided_slice/stack_1:output:01dense_features/V23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V23/strided_slice
"dense_features/V23/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V23/Reshape/shape/1Ò
 dense_features/V23/Reshape/shapePack)dense_features/V23/strided_slice:output:0+dense_features/V23/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V23/Reshape/shape¬
dense_features/V23/ReshapeReshape
inputs_v23)dense_features/V23/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V23/Reshapen
dense_features/V24/ShapeShape
inputs_v24*
T0*
_output_shapes
:2
dense_features/V24/Shape
&dense_features/V24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V24/strided_slice/stack
(dense_features/V24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V24/strided_slice/stack_1
(dense_features/V24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V24/strided_slice/stack_2Ô
 dense_features/V24/strided_sliceStridedSlice!dense_features/V24/Shape:output:0/dense_features/V24/strided_slice/stack:output:01dense_features/V24/strided_slice/stack_1:output:01dense_features/V24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V24/strided_slice
"dense_features/V24/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V24/Reshape/shape/1Ò
 dense_features/V24/Reshape/shapePack)dense_features/V24/strided_slice:output:0+dense_features/V24/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V24/Reshape/shape¬
dense_features/V24/ReshapeReshape
inputs_v24)dense_features/V24/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V24/Reshapen
dense_features/V25/ShapeShape
inputs_v25*
T0*
_output_shapes
:2
dense_features/V25/Shape
&dense_features/V25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V25/strided_slice/stack
(dense_features/V25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V25/strided_slice/stack_1
(dense_features/V25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V25/strided_slice/stack_2Ô
 dense_features/V25/strided_sliceStridedSlice!dense_features/V25/Shape:output:0/dense_features/V25/strided_slice/stack:output:01dense_features/V25/strided_slice/stack_1:output:01dense_features/V25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V25/strided_slice
"dense_features/V25/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V25/Reshape/shape/1Ò
 dense_features/V25/Reshape/shapePack)dense_features/V25/strided_slice:output:0+dense_features/V25/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V25/Reshape/shape¬
dense_features/V25/ReshapeReshape
inputs_v25)dense_features/V25/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V25/Reshapen
dense_features/V26/ShapeShape
inputs_v26*
T0*
_output_shapes
:2
dense_features/V26/Shape
&dense_features/V26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V26/strided_slice/stack
(dense_features/V26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V26/strided_slice/stack_1
(dense_features/V26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V26/strided_slice/stack_2Ô
 dense_features/V26/strided_sliceStridedSlice!dense_features/V26/Shape:output:0/dense_features/V26/strided_slice/stack:output:01dense_features/V26/strided_slice/stack_1:output:01dense_features/V26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V26/strided_slice
"dense_features/V26/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V26/Reshape/shape/1Ò
 dense_features/V26/Reshape/shapePack)dense_features/V26/strided_slice:output:0+dense_features/V26/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V26/Reshape/shape¬
dense_features/V26/ReshapeReshape
inputs_v26)dense_features/V26/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V26/Reshapen
dense_features/V27/ShapeShape
inputs_v27*
T0*
_output_shapes
:2
dense_features/V27/Shape
&dense_features/V27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V27/strided_slice/stack
(dense_features/V27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V27/strided_slice/stack_1
(dense_features/V27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V27/strided_slice/stack_2Ô
 dense_features/V27/strided_sliceStridedSlice!dense_features/V27/Shape:output:0/dense_features/V27/strided_slice/stack:output:01dense_features/V27/strided_slice/stack_1:output:01dense_features/V27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V27/strided_slice
"dense_features/V27/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V27/Reshape/shape/1Ò
 dense_features/V27/Reshape/shapePack)dense_features/V27/strided_slice:output:0+dense_features/V27/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V27/Reshape/shape¬
dense_features/V27/ReshapeReshape
inputs_v27)dense_features/V27/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V27/Reshapen
dense_features/V28/ShapeShape
inputs_v28*
T0*
_output_shapes
:2
dense_features/V28/Shape
&dense_features/V28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&dense_features/V28/strided_slice/stack
(dense_features/V28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V28/strided_slice/stack_1
(dense_features/V28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(dense_features/V28/strided_slice/stack_2Ô
 dense_features/V28/strided_sliceStridedSlice!dense_features/V28/Shape:output:0/dense_features/V28/strided_slice/stack:output:01dense_features/V28/strided_slice/stack_1:output:01dense_features/V28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 dense_features/V28/strided_slice
"dense_features/V28/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"dense_features/V28/Reshape/shape/1Ò
 dense_features/V28/Reshape/shapePack)dense_features/V28/strided_slice:output:0+dense_features/V28/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 dense_features/V28/Reshape/shape¬
dense_features/V28/ReshapeReshape
inputs_v28)dense_features/V28/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V28/Reshapek
dense_features/V3/ShapeShape	inputs_v3*
T0*
_output_shapes
:2
dense_features/V3/Shape
%dense_features/V3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%dense_features/V3/strided_slice/stack
'dense_features/V3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V3/strided_slice/stack_1
'dense_features/V3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V3/strided_slice/stack_2Î
dense_features/V3/strided_sliceStridedSlice dense_features/V3/Shape:output:0.dense_features/V3/strided_slice/stack:output:00dense_features/V3/strided_slice/stack_1:output:00dense_features/V3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
dense_features/V3/strided_slice
!dense_features/V3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!dense_features/V3/Reshape/shape/1Î
dense_features/V3/Reshape/shapePack(dense_features/V3/strided_slice:output:0*dense_features/V3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
dense_features/V3/Reshape/shape¨
dense_features/V3/ReshapeReshape	inputs_v3(dense_features/V3/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V3/Reshapek
dense_features/V4/ShapeShape	inputs_v4*
T0*
_output_shapes
:2
dense_features/V4/Shape
%dense_features/V4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%dense_features/V4/strided_slice/stack
'dense_features/V4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V4/strided_slice/stack_1
'dense_features/V4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V4/strided_slice/stack_2Î
dense_features/V4/strided_sliceStridedSlice dense_features/V4/Shape:output:0.dense_features/V4/strided_slice/stack:output:00dense_features/V4/strided_slice/stack_1:output:00dense_features/V4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
dense_features/V4/strided_slice
!dense_features/V4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!dense_features/V4/Reshape/shape/1Î
dense_features/V4/Reshape/shapePack(dense_features/V4/strided_slice:output:0*dense_features/V4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
dense_features/V4/Reshape/shape¨
dense_features/V4/ReshapeReshape	inputs_v4(dense_features/V4/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V4/Reshapek
dense_features/V5/ShapeShape	inputs_v5*
T0*
_output_shapes
:2
dense_features/V5/Shape
%dense_features/V5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%dense_features/V5/strided_slice/stack
'dense_features/V5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V5/strided_slice/stack_1
'dense_features/V5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V5/strided_slice/stack_2Î
dense_features/V5/strided_sliceStridedSlice dense_features/V5/Shape:output:0.dense_features/V5/strided_slice/stack:output:00dense_features/V5/strided_slice/stack_1:output:00dense_features/V5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
dense_features/V5/strided_slice
!dense_features/V5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!dense_features/V5/Reshape/shape/1Î
dense_features/V5/Reshape/shapePack(dense_features/V5/strided_slice:output:0*dense_features/V5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
dense_features/V5/Reshape/shape¨
dense_features/V5/ReshapeReshape	inputs_v5(dense_features/V5/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V5/Reshapek
dense_features/V6/ShapeShape	inputs_v6*
T0*
_output_shapes
:2
dense_features/V6/Shape
%dense_features/V6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%dense_features/V6/strided_slice/stack
'dense_features/V6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V6/strided_slice/stack_1
'dense_features/V6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V6/strided_slice/stack_2Î
dense_features/V6/strided_sliceStridedSlice dense_features/V6/Shape:output:0.dense_features/V6/strided_slice/stack:output:00dense_features/V6/strided_slice/stack_1:output:00dense_features/V6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
dense_features/V6/strided_slice
!dense_features/V6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!dense_features/V6/Reshape/shape/1Î
dense_features/V6/Reshape/shapePack(dense_features/V6/strided_slice:output:0*dense_features/V6/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
dense_features/V6/Reshape/shape¨
dense_features/V6/ReshapeReshape	inputs_v6(dense_features/V6/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V6/Reshapek
dense_features/V7/ShapeShape	inputs_v7*
T0*
_output_shapes
:2
dense_features/V7/Shape
%dense_features/V7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%dense_features/V7/strided_slice/stack
'dense_features/V7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V7/strided_slice/stack_1
'dense_features/V7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V7/strided_slice/stack_2Î
dense_features/V7/strided_sliceStridedSlice dense_features/V7/Shape:output:0.dense_features/V7/strided_slice/stack:output:00dense_features/V7/strided_slice/stack_1:output:00dense_features/V7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
dense_features/V7/strided_slice
!dense_features/V7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!dense_features/V7/Reshape/shape/1Î
dense_features/V7/Reshape/shapePack(dense_features/V7/strided_slice:output:0*dense_features/V7/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
dense_features/V7/Reshape/shape¨
dense_features/V7/ReshapeReshape	inputs_v7(dense_features/V7/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V7/Reshapek
dense_features/V8/ShapeShape	inputs_v8*
T0*
_output_shapes
:2
dense_features/V8/Shape
%dense_features/V8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%dense_features/V8/strided_slice/stack
'dense_features/V8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V8/strided_slice/stack_1
'dense_features/V8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V8/strided_slice/stack_2Î
dense_features/V8/strided_sliceStridedSlice dense_features/V8/Shape:output:0.dense_features/V8/strided_slice/stack:output:00dense_features/V8/strided_slice/stack_1:output:00dense_features/V8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
dense_features/V8/strided_slice
!dense_features/V8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!dense_features/V8/Reshape/shape/1Î
dense_features/V8/Reshape/shapePack(dense_features/V8/strided_slice:output:0*dense_features/V8/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
dense_features/V8/Reshape/shape¨
dense_features/V8/ReshapeReshape	inputs_v8(dense_features/V8/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V8/Reshapek
dense_features/V9/ShapeShape	inputs_v9*
T0*
_output_shapes
:2
dense_features/V9/Shape
%dense_features/V9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%dense_features/V9/strided_slice/stack
'dense_features/V9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V9/strided_slice/stack_1
'dense_features/V9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'dense_features/V9/strided_slice/stack_2Î
dense_features/V9/strided_sliceStridedSlice dense_features/V9/Shape:output:0.dense_features/V9/strided_slice/stack:output:00dense_features/V9/strided_slice/stack_1:output:00dense_features/V9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
dense_features/V9/strided_slice
!dense_features/V9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!dense_features/V9/Reshape/shape/1Î
dense_features/V9/Reshape/shapePack(dense_features/V9/strided_slice:output:0*dense_features/V9/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
dense_features/V9/Reshape/shape¨
dense_features/V9/ReshapeReshape	inputs_v9(dense_features/V9/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/V9/Reshape
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
dense_features/concat/axisë	
dense_features/concatConcatV2&dense_features/Amount/Reshape:output:0$dense_features/Time/Reshape:output:0"dense_features/V1/Reshape:output:0#dense_features/V10/Reshape:output:0#dense_features/V11/Reshape:output:0#dense_features/V12/Reshape:output:0#dense_features/V13/Reshape:output:0#dense_features/V14/Reshape:output:0#dense_features/V15/Reshape:output:0#dense_features/V16/Reshape:output:0#dense_features/V17/Reshape:output:0#dense_features/V18/Reshape:output:0#dense_features/V19/Reshape:output:0"dense_features/V2/Reshape:output:0#dense_features/V20/Reshape:output:0#dense_features/V21/Reshape:output:0#dense_features/V22/Reshape:output:0#dense_features/V23/Reshape:output:0#dense_features/V24/Reshape:output:0#dense_features/V25/Reshape:output:0#dense_features/V26/Reshape:output:0#dense_features/V27/Reshape:output:0#dense_features/V28/Reshape:output:0"dense_features/V3/Reshape:output:0"dense_features/V4/Reshape:output:0"dense_features/V5/Reshape:output:0"dense_features/V6/Reshape:output:0"dense_features/V7/Reshape:output:0"dense_features/V8/Reshape:output:0"dense_features/V9/Reshape:output:0#dense_features/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/concatÎ
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization/batchnorm/ReadVariableOp
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2%
#batch_normalization/batchnorm/add/yØ
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/add
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/RsqrtÚ
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOpÕ
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/mulÊ
#batch_normalization/batchnorm/mul_1Muldense_features/concat:output:0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#batch_normalization/batchnorm/mul_1Ô
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1Õ
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/mul_2Ô
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2Ó
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/subÕ
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#batch_normalization/batchnorm/add_1
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp¦
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAdds
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Softmaxk
IdentityIdentitydense/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*ç
_input_shapesÕ
Ò:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::::::V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameinputs/Amount:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinputs/Time:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V1:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V10:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V11:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V12:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V13:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V14:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V15:S	O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V16:S
O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V17:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V18:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V19:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V2:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V20:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V21:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V22:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V23:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V24:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V25:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V26:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V27:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V28:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V3:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V4:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V5:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V6:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V7:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V8:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V9
ò%

,__inference_functional_1_layer_call_fn_83354
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
	inputs_v9
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¡
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
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
 !"#*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_825332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*ç
_input_shapesÕ
Ò:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameinputs/Amount:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinputs/Time:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V1:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V10:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V11:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V12:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V13:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V14:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V15:S	O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V16:S
O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V17:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V18:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V19:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V2:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V20:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V21:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V22:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V23:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V24:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V25:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V26:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V27:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/V28:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V3:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V4:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V5:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V6:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V7:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V8:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/V9
©
Ë
I__inference_dense_features_layer_call_and_return_conditional_losses_82271
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
identityT
Amount/ShapeShapefeatures*
T0*
_output_shapes
:2
Amount/Shape
Amount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Amount/strided_slice/stack
Amount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Amount/strided_slice/stack_1
Amount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Amount/strided_slice/stack_2
Amount/strided_sliceStridedSliceAmount/Shape:output:0#Amount/strided_slice/stack:output:0%Amount/strided_slice/stack_1:output:0%Amount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Amount/strided_slicer
Amount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Amount/Reshape/shape/1¢
Amount/Reshape/shapePackAmount/strided_slice:output:0Amount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Amount/Reshape/shape
Amount/ReshapeReshapefeaturesAmount/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Amount/ReshapeR

Time/ShapeShape
features_1*
T0*
_output_shapes
:2

Time/Shape~
Time/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Time/strided_slice/stack
Time/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Time/strided_slice/stack_1
Time/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Time/strided_slice/stack_2
Time/strided_sliceStridedSliceTime/Shape:output:0!Time/strided_slice/stack:output:0#Time/strided_slice/stack_1:output:0#Time/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Time/strided_slicen
Time/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Time/Reshape/shape/1
Time/Reshape/shapePackTime/strided_slice:output:0Time/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Time/Reshape/shape
Time/ReshapeReshape
features_1Time/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Time/ReshapeN
V1/ShapeShape
features_2*
T0*
_output_shapes
:2

V1/Shapez
V1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V1/strided_slice/stack~
V1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V1/strided_slice/stack_1~
V1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V1/strided_slice/stack_2ô
V1/strided_sliceStridedSliceV1/Shape:output:0V1/strided_slice/stack:output:0!V1/strided_slice/stack_1:output:0!V1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V1/strided_slicej
V1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V1/Reshape/shape/1
V1/Reshape/shapePackV1/strided_slice:output:0V1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V1/Reshape/shape|

V1/ReshapeReshape
features_2V1/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V1/ReshapeP
	V10/ShapeShape
features_3*
T0*
_output_shapes
:2
	V10/Shape|
V10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V10/strided_slice/stack
V10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V10/strided_slice/stack_1
V10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V10/strided_slice/stack_2ú
V10/strided_sliceStridedSliceV10/Shape:output:0 V10/strided_slice/stack:output:0"V10/strided_slice/stack_1:output:0"V10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V10/strided_slicel
V10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V10/Reshape/shape/1
V10/Reshape/shapePackV10/strided_slice:output:0V10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V10/Reshape/shape
V10/ReshapeReshape
features_3V10/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V10/ReshapeP
	V11/ShapeShape
features_4*
T0*
_output_shapes
:2
	V11/Shape|
V11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V11/strided_slice/stack
V11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V11/strided_slice/stack_1
V11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V11/strided_slice/stack_2ú
V11/strided_sliceStridedSliceV11/Shape:output:0 V11/strided_slice/stack:output:0"V11/strided_slice/stack_1:output:0"V11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V11/strided_slicel
V11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V11/Reshape/shape/1
V11/Reshape/shapePackV11/strided_slice:output:0V11/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V11/Reshape/shape
V11/ReshapeReshape
features_4V11/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V11/ReshapeP
	V12/ShapeShape
features_5*
T0*
_output_shapes
:2
	V12/Shape|
V12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V12/strided_slice/stack
V12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V12/strided_slice/stack_1
V12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V12/strided_slice/stack_2ú
V12/strided_sliceStridedSliceV12/Shape:output:0 V12/strided_slice/stack:output:0"V12/strided_slice/stack_1:output:0"V12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V12/strided_slicel
V12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V12/Reshape/shape/1
V12/Reshape/shapePackV12/strided_slice:output:0V12/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V12/Reshape/shape
V12/ReshapeReshape
features_5V12/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V12/ReshapeP
	V13/ShapeShape
features_6*
T0*
_output_shapes
:2
	V13/Shape|
V13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V13/strided_slice/stack
V13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V13/strided_slice/stack_1
V13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V13/strided_slice/stack_2ú
V13/strided_sliceStridedSliceV13/Shape:output:0 V13/strided_slice/stack:output:0"V13/strided_slice/stack_1:output:0"V13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V13/strided_slicel
V13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V13/Reshape/shape/1
V13/Reshape/shapePackV13/strided_slice:output:0V13/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V13/Reshape/shape
V13/ReshapeReshape
features_6V13/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V13/ReshapeP
	V14/ShapeShape
features_7*
T0*
_output_shapes
:2
	V14/Shape|
V14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V14/strided_slice/stack
V14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V14/strided_slice/stack_1
V14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V14/strided_slice/stack_2ú
V14/strided_sliceStridedSliceV14/Shape:output:0 V14/strided_slice/stack:output:0"V14/strided_slice/stack_1:output:0"V14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V14/strided_slicel
V14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V14/Reshape/shape/1
V14/Reshape/shapePackV14/strided_slice:output:0V14/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V14/Reshape/shape
V14/ReshapeReshape
features_7V14/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V14/ReshapeP
	V15/ShapeShape
features_8*
T0*
_output_shapes
:2
	V15/Shape|
V15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V15/strided_slice/stack
V15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V15/strided_slice/stack_1
V15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V15/strided_slice/stack_2ú
V15/strided_sliceStridedSliceV15/Shape:output:0 V15/strided_slice/stack:output:0"V15/strided_slice/stack_1:output:0"V15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V15/strided_slicel
V15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V15/Reshape/shape/1
V15/Reshape/shapePackV15/strided_slice:output:0V15/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V15/Reshape/shape
V15/ReshapeReshape
features_8V15/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V15/ReshapeP
	V16/ShapeShape
features_9*
T0*
_output_shapes
:2
	V16/Shape|
V16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V16/strided_slice/stack
V16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V16/strided_slice/stack_1
V16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V16/strided_slice/stack_2ú
V16/strided_sliceStridedSliceV16/Shape:output:0 V16/strided_slice/stack:output:0"V16/strided_slice/stack_1:output:0"V16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V16/strided_slicel
V16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V16/Reshape/shape/1
V16/Reshape/shapePackV16/strided_slice:output:0V16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V16/Reshape/shape
V16/ReshapeReshape
features_9V16/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V16/ReshapeQ
	V17/ShapeShapefeatures_10*
T0*
_output_shapes
:2
	V17/Shape|
V17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V17/strided_slice/stack
V17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V17/strided_slice/stack_1
V17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V17/strided_slice/stack_2ú
V17/strided_sliceStridedSliceV17/Shape:output:0 V17/strided_slice/stack:output:0"V17/strided_slice/stack_1:output:0"V17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V17/strided_slicel
V17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V17/Reshape/shape/1
V17/Reshape/shapePackV17/strided_slice:output:0V17/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V17/Reshape/shape
V17/ReshapeReshapefeatures_10V17/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V17/ReshapeQ
	V18/ShapeShapefeatures_11*
T0*
_output_shapes
:2
	V18/Shape|
V18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V18/strided_slice/stack
V18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V18/strided_slice/stack_1
V18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V18/strided_slice/stack_2ú
V18/strided_sliceStridedSliceV18/Shape:output:0 V18/strided_slice/stack:output:0"V18/strided_slice/stack_1:output:0"V18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V18/strided_slicel
V18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V18/Reshape/shape/1
V18/Reshape/shapePackV18/strided_slice:output:0V18/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V18/Reshape/shape
V18/ReshapeReshapefeatures_11V18/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V18/ReshapeQ
	V19/ShapeShapefeatures_12*
T0*
_output_shapes
:2
	V19/Shape|
V19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V19/strided_slice/stack
V19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V19/strided_slice/stack_1
V19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V19/strided_slice/stack_2ú
V19/strided_sliceStridedSliceV19/Shape:output:0 V19/strided_slice/stack:output:0"V19/strided_slice/stack_1:output:0"V19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V19/strided_slicel
V19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V19/Reshape/shape/1
V19/Reshape/shapePackV19/strided_slice:output:0V19/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V19/Reshape/shape
V19/ReshapeReshapefeatures_12V19/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V19/ReshapeO
V2/ShapeShapefeatures_13*
T0*
_output_shapes
:2

V2/Shapez
V2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V2/strided_slice/stack~
V2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V2/strided_slice/stack_1~
V2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V2/strided_slice/stack_2ô
V2/strided_sliceStridedSliceV2/Shape:output:0V2/strided_slice/stack:output:0!V2/strided_slice/stack_1:output:0!V2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V2/strided_slicej
V2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V2/Reshape/shape/1
V2/Reshape/shapePackV2/strided_slice:output:0V2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V2/Reshape/shape}

V2/ReshapeReshapefeatures_13V2/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V2/ReshapeQ
	V20/ShapeShapefeatures_14*
T0*
_output_shapes
:2
	V20/Shape|
V20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V20/strided_slice/stack
V20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V20/strided_slice/stack_1
V20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V20/strided_slice/stack_2ú
V20/strided_sliceStridedSliceV20/Shape:output:0 V20/strided_slice/stack:output:0"V20/strided_slice/stack_1:output:0"V20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V20/strided_slicel
V20/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V20/Reshape/shape/1
V20/Reshape/shapePackV20/strided_slice:output:0V20/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V20/Reshape/shape
V20/ReshapeReshapefeatures_14V20/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V20/ReshapeQ
	V21/ShapeShapefeatures_15*
T0*
_output_shapes
:2
	V21/Shape|
V21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V21/strided_slice/stack
V21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V21/strided_slice/stack_1
V21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V21/strided_slice/stack_2ú
V21/strided_sliceStridedSliceV21/Shape:output:0 V21/strided_slice/stack:output:0"V21/strided_slice/stack_1:output:0"V21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V21/strided_slicel
V21/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V21/Reshape/shape/1
V21/Reshape/shapePackV21/strided_slice:output:0V21/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V21/Reshape/shape
V21/ReshapeReshapefeatures_15V21/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V21/ReshapeQ
	V22/ShapeShapefeatures_16*
T0*
_output_shapes
:2
	V22/Shape|
V22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V22/strided_slice/stack
V22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V22/strided_slice/stack_1
V22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V22/strided_slice/stack_2ú
V22/strided_sliceStridedSliceV22/Shape:output:0 V22/strided_slice/stack:output:0"V22/strided_slice/stack_1:output:0"V22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V22/strided_slicel
V22/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V22/Reshape/shape/1
V22/Reshape/shapePackV22/strided_slice:output:0V22/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V22/Reshape/shape
V22/ReshapeReshapefeatures_16V22/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V22/ReshapeQ
	V23/ShapeShapefeatures_17*
T0*
_output_shapes
:2
	V23/Shape|
V23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V23/strided_slice/stack
V23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V23/strided_slice/stack_1
V23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V23/strided_slice/stack_2ú
V23/strided_sliceStridedSliceV23/Shape:output:0 V23/strided_slice/stack:output:0"V23/strided_slice/stack_1:output:0"V23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V23/strided_slicel
V23/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V23/Reshape/shape/1
V23/Reshape/shapePackV23/strided_slice:output:0V23/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V23/Reshape/shape
V23/ReshapeReshapefeatures_17V23/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V23/ReshapeQ
	V24/ShapeShapefeatures_18*
T0*
_output_shapes
:2
	V24/Shape|
V24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V24/strided_slice/stack
V24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V24/strided_slice/stack_1
V24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V24/strided_slice/stack_2ú
V24/strided_sliceStridedSliceV24/Shape:output:0 V24/strided_slice/stack:output:0"V24/strided_slice/stack_1:output:0"V24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V24/strided_slicel
V24/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V24/Reshape/shape/1
V24/Reshape/shapePackV24/strided_slice:output:0V24/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V24/Reshape/shape
V24/ReshapeReshapefeatures_18V24/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V24/ReshapeQ
	V25/ShapeShapefeatures_19*
T0*
_output_shapes
:2
	V25/Shape|
V25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V25/strided_slice/stack
V25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V25/strided_slice/stack_1
V25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V25/strided_slice/stack_2ú
V25/strided_sliceStridedSliceV25/Shape:output:0 V25/strided_slice/stack:output:0"V25/strided_slice/stack_1:output:0"V25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V25/strided_slicel
V25/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V25/Reshape/shape/1
V25/Reshape/shapePackV25/strided_slice:output:0V25/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V25/Reshape/shape
V25/ReshapeReshapefeatures_19V25/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V25/ReshapeQ
	V26/ShapeShapefeatures_20*
T0*
_output_shapes
:2
	V26/Shape|
V26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V26/strided_slice/stack
V26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V26/strided_slice/stack_1
V26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V26/strided_slice/stack_2ú
V26/strided_sliceStridedSliceV26/Shape:output:0 V26/strided_slice/stack:output:0"V26/strided_slice/stack_1:output:0"V26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V26/strided_slicel
V26/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V26/Reshape/shape/1
V26/Reshape/shapePackV26/strided_slice:output:0V26/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V26/Reshape/shape
V26/ReshapeReshapefeatures_20V26/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V26/ReshapeQ
	V27/ShapeShapefeatures_21*
T0*
_output_shapes
:2
	V27/Shape|
V27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V27/strided_slice/stack
V27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V27/strided_slice/stack_1
V27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V27/strided_slice/stack_2ú
V27/strided_sliceStridedSliceV27/Shape:output:0 V27/strided_slice/stack:output:0"V27/strided_slice/stack_1:output:0"V27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V27/strided_slicel
V27/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V27/Reshape/shape/1
V27/Reshape/shapePackV27/strided_slice:output:0V27/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V27/Reshape/shape
V27/ReshapeReshapefeatures_21V27/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V27/ReshapeQ
	V28/ShapeShapefeatures_22*
T0*
_output_shapes
:2
	V28/Shape|
V28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V28/strided_slice/stack
V28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V28/strided_slice/stack_1
V28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V28/strided_slice/stack_2ú
V28/strided_sliceStridedSliceV28/Shape:output:0 V28/strided_slice/stack:output:0"V28/strided_slice/stack_1:output:0"V28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V28/strided_slicel
V28/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V28/Reshape/shape/1
V28/Reshape/shapePackV28/strided_slice:output:0V28/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V28/Reshape/shape
V28/ReshapeReshapefeatures_22V28/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V28/ReshapeO
V3/ShapeShapefeatures_23*
T0*
_output_shapes
:2

V3/Shapez
V3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V3/strided_slice/stack~
V3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V3/strided_slice/stack_1~
V3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V3/strided_slice/stack_2ô
V3/strided_sliceStridedSliceV3/Shape:output:0V3/strided_slice/stack:output:0!V3/strided_slice/stack_1:output:0!V3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V3/strided_slicej
V3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V3/Reshape/shape/1
V3/Reshape/shapePackV3/strided_slice:output:0V3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V3/Reshape/shape}

V3/ReshapeReshapefeatures_23V3/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V3/ReshapeO
V4/ShapeShapefeatures_24*
T0*
_output_shapes
:2

V4/Shapez
V4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V4/strided_slice/stack~
V4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V4/strided_slice/stack_1~
V4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V4/strided_slice/stack_2ô
V4/strided_sliceStridedSliceV4/Shape:output:0V4/strided_slice/stack:output:0!V4/strided_slice/stack_1:output:0!V4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V4/strided_slicej
V4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V4/Reshape/shape/1
V4/Reshape/shapePackV4/strided_slice:output:0V4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V4/Reshape/shape}

V4/ReshapeReshapefeatures_24V4/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V4/ReshapeO
V5/ShapeShapefeatures_25*
T0*
_output_shapes
:2

V5/Shapez
V5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V5/strided_slice/stack~
V5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V5/strided_slice/stack_1~
V5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V5/strided_slice/stack_2ô
V5/strided_sliceStridedSliceV5/Shape:output:0V5/strided_slice/stack:output:0!V5/strided_slice/stack_1:output:0!V5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V5/strided_slicej
V5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V5/Reshape/shape/1
V5/Reshape/shapePackV5/strided_slice:output:0V5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V5/Reshape/shape}

V5/ReshapeReshapefeatures_25V5/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V5/ReshapeO
V6/ShapeShapefeatures_26*
T0*
_output_shapes
:2

V6/Shapez
V6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V6/strided_slice/stack~
V6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V6/strided_slice/stack_1~
V6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V6/strided_slice/stack_2ô
V6/strided_sliceStridedSliceV6/Shape:output:0V6/strided_slice/stack:output:0!V6/strided_slice/stack_1:output:0!V6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V6/strided_slicej
V6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V6/Reshape/shape/1
V6/Reshape/shapePackV6/strided_slice:output:0V6/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V6/Reshape/shape}

V6/ReshapeReshapefeatures_26V6/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V6/ReshapeO
V7/ShapeShapefeatures_27*
T0*
_output_shapes
:2

V7/Shapez
V7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V7/strided_slice/stack~
V7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V7/strided_slice/stack_1~
V7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V7/strided_slice/stack_2ô
V7/strided_sliceStridedSliceV7/Shape:output:0V7/strided_slice/stack:output:0!V7/strided_slice/stack_1:output:0!V7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V7/strided_slicej
V7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V7/Reshape/shape/1
V7/Reshape/shapePackV7/strided_slice:output:0V7/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V7/Reshape/shape}

V7/ReshapeReshapefeatures_27V7/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V7/ReshapeO
V8/ShapeShapefeatures_28*
T0*
_output_shapes
:2

V8/Shapez
V8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V8/strided_slice/stack~
V8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V8/strided_slice/stack_1~
V8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V8/strided_slice/stack_2ô
V8/strided_sliceStridedSliceV8/Shape:output:0V8/strided_slice/stack:output:0!V8/strided_slice/stack_1:output:0!V8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V8/strided_slicej
V8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V8/Reshape/shape/1
V8/Reshape/shapePackV8/strided_slice:output:0V8/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V8/Reshape/shape}

V8/ReshapeReshapefeatures_28V8/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V8/ReshapeO
V9/ShapeShapefeatures_29*
T0*
_output_shapes
:2

V9/Shapez
V9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V9/strided_slice/stack~
V9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V9/strided_slice/stack_1~
V9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V9/strided_slice/stack_2ô
V9/strided_sliceStridedSliceV9/Shape:output:0V9/strided_slice/stack:output:0!V9/strided_slice/stack_1:output:0!V9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V9/strided_slicej
V9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V9/Reshape/shape/1
V9/Reshape/shapePackV9/strided_slice:output:0V9/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V9/Reshape/shape}

V9/ReshapeReshapefeatures_29V9/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V9/Reshapee
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
concat/axisü
concatConcatV2Amount/Reshape:output:0Time/Reshape:output:0V1/Reshape:output:0V10/Reshape:output:0V11/Reshape:output:0V12/Reshape:output:0V13/Reshape:output:0V14/Reshape:output:0V15/Reshape:output:0V16/Reshape:output:0V17/Reshape:output:0V18/Reshape:output:0V19/Reshape:output:0V2/Reshape:output:0V20/Reshape:output:0V21/Reshape:output:0V22/Reshape:output:0V23/Reshape:output:0V24/Reshape:output:0V25/Reshape:output:0V26/Reshape:output:0V27/Reshape:output:0V28/Reshape:output:0V3/Reshape:output:0V4/Reshape:output:0V5/Reshape:output:0V6/Reshape:output:0V7/Reshape:output:0V8/Reshape:output:0V9/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ï
_input_shapes½
º:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:Q	M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:Q
M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
features
ÿ-
ü
G__inference_functional_1_layer_call_and_return_conditional_losses_82533

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
	inputs_29
batch_normalization_82518
batch_normalization_82520
batch_normalization_82522
batch_normalization_82524
dense_82527
dense_82529
identity¢+batch_normalization/StatefulPartitionedCall¢dense/StatefulPartitionedCall¸
dense_features/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16	inputs_17	inputs_18	inputs_19	inputs_20	inputs_21	inputs_22	inputs_23	inputs_24	inputs_25	inputs_26	inputs_27	inputs_28	inputs_29*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_819962 
dense_features/PartitionedCall¡
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0batch_normalization_82518batch_normalization_82520batch_normalization_82522batch_normalization_82524*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_816442-
+batch_normalization/StatefulPartitionedCall°
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_82527dense_82529*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_823882
dense/StatefulPartitionedCallÈ
IdentityIdentity&dense/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*ç
_input_shapesÕ
Ò:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:O	K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:O
K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
ð
I__inference_dense_features_layer_call_and_return_conditional_losses_83950
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
identity[
Amount/ShapeShapefeatures_amount*
T0*
_output_shapes
:2
Amount/Shape
Amount/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Amount/strided_slice/stack
Amount/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Amount/strided_slice/stack_1
Amount/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Amount/strided_slice/stack_2
Amount/strided_sliceStridedSliceAmount/Shape:output:0#Amount/strided_slice/stack:output:0%Amount/strided_slice/stack_1:output:0%Amount/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Amount/strided_slicer
Amount/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Amount/Reshape/shape/1¢
Amount/Reshape/shapePackAmount/strided_slice:output:0Amount/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Amount/Reshape/shape
Amount/ReshapeReshapefeatures_amountAmount/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Amount/ReshapeU

Time/ShapeShapefeatures_time*
T0*
_output_shapes
:2

Time/Shape~
Time/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Time/strided_slice/stack
Time/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Time/strided_slice/stack_1
Time/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Time/strided_slice/stack_2
Time/strided_sliceStridedSliceTime/Shape:output:0!Time/strided_slice/stack:output:0#Time/strided_slice/stack_1:output:0#Time/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Time/strided_slicen
Time/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Time/Reshape/shape/1
Time/Reshape/shapePackTime/strided_slice:output:0Time/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Time/Reshape/shape
Time/ReshapeReshapefeatures_timeTime/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Time/ReshapeO
V1/ShapeShapefeatures_v1*
T0*
_output_shapes
:2

V1/Shapez
V1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V1/strided_slice/stack~
V1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V1/strided_slice/stack_1~
V1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V1/strided_slice/stack_2ô
V1/strided_sliceStridedSliceV1/Shape:output:0V1/strided_slice/stack:output:0!V1/strided_slice/stack_1:output:0!V1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V1/strided_slicej
V1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V1/Reshape/shape/1
V1/Reshape/shapePackV1/strided_slice:output:0V1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V1/Reshape/shape}

V1/ReshapeReshapefeatures_v1V1/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V1/ReshapeR
	V10/ShapeShapefeatures_v10*
T0*
_output_shapes
:2
	V10/Shape|
V10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V10/strided_slice/stack
V10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V10/strided_slice/stack_1
V10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V10/strided_slice/stack_2ú
V10/strided_sliceStridedSliceV10/Shape:output:0 V10/strided_slice/stack:output:0"V10/strided_slice/stack_1:output:0"V10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V10/strided_slicel
V10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V10/Reshape/shape/1
V10/Reshape/shapePackV10/strided_slice:output:0V10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V10/Reshape/shape
V10/ReshapeReshapefeatures_v10V10/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V10/ReshapeR
	V11/ShapeShapefeatures_v11*
T0*
_output_shapes
:2
	V11/Shape|
V11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V11/strided_slice/stack
V11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V11/strided_slice/stack_1
V11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V11/strided_slice/stack_2ú
V11/strided_sliceStridedSliceV11/Shape:output:0 V11/strided_slice/stack:output:0"V11/strided_slice/stack_1:output:0"V11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V11/strided_slicel
V11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V11/Reshape/shape/1
V11/Reshape/shapePackV11/strided_slice:output:0V11/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V11/Reshape/shape
V11/ReshapeReshapefeatures_v11V11/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V11/ReshapeR
	V12/ShapeShapefeatures_v12*
T0*
_output_shapes
:2
	V12/Shape|
V12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V12/strided_slice/stack
V12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V12/strided_slice/stack_1
V12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V12/strided_slice/stack_2ú
V12/strided_sliceStridedSliceV12/Shape:output:0 V12/strided_slice/stack:output:0"V12/strided_slice/stack_1:output:0"V12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V12/strided_slicel
V12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V12/Reshape/shape/1
V12/Reshape/shapePackV12/strided_slice:output:0V12/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V12/Reshape/shape
V12/ReshapeReshapefeatures_v12V12/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V12/ReshapeR
	V13/ShapeShapefeatures_v13*
T0*
_output_shapes
:2
	V13/Shape|
V13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V13/strided_slice/stack
V13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V13/strided_slice/stack_1
V13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V13/strided_slice/stack_2ú
V13/strided_sliceStridedSliceV13/Shape:output:0 V13/strided_slice/stack:output:0"V13/strided_slice/stack_1:output:0"V13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V13/strided_slicel
V13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V13/Reshape/shape/1
V13/Reshape/shapePackV13/strided_slice:output:0V13/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V13/Reshape/shape
V13/ReshapeReshapefeatures_v13V13/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V13/ReshapeR
	V14/ShapeShapefeatures_v14*
T0*
_output_shapes
:2
	V14/Shape|
V14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V14/strided_slice/stack
V14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V14/strided_slice/stack_1
V14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V14/strided_slice/stack_2ú
V14/strided_sliceStridedSliceV14/Shape:output:0 V14/strided_slice/stack:output:0"V14/strided_slice/stack_1:output:0"V14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V14/strided_slicel
V14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V14/Reshape/shape/1
V14/Reshape/shapePackV14/strided_slice:output:0V14/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V14/Reshape/shape
V14/ReshapeReshapefeatures_v14V14/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V14/ReshapeR
	V15/ShapeShapefeatures_v15*
T0*
_output_shapes
:2
	V15/Shape|
V15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V15/strided_slice/stack
V15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V15/strided_slice/stack_1
V15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V15/strided_slice/stack_2ú
V15/strided_sliceStridedSliceV15/Shape:output:0 V15/strided_slice/stack:output:0"V15/strided_slice/stack_1:output:0"V15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V15/strided_slicel
V15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V15/Reshape/shape/1
V15/Reshape/shapePackV15/strided_slice:output:0V15/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V15/Reshape/shape
V15/ReshapeReshapefeatures_v15V15/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V15/ReshapeR
	V16/ShapeShapefeatures_v16*
T0*
_output_shapes
:2
	V16/Shape|
V16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V16/strided_slice/stack
V16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V16/strided_slice/stack_1
V16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V16/strided_slice/stack_2ú
V16/strided_sliceStridedSliceV16/Shape:output:0 V16/strided_slice/stack:output:0"V16/strided_slice/stack_1:output:0"V16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V16/strided_slicel
V16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V16/Reshape/shape/1
V16/Reshape/shapePackV16/strided_slice:output:0V16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V16/Reshape/shape
V16/ReshapeReshapefeatures_v16V16/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V16/ReshapeR
	V17/ShapeShapefeatures_v17*
T0*
_output_shapes
:2
	V17/Shape|
V17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V17/strided_slice/stack
V17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V17/strided_slice/stack_1
V17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V17/strided_slice/stack_2ú
V17/strided_sliceStridedSliceV17/Shape:output:0 V17/strided_slice/stack:output:0"V17/strided_slice/stack_1:output:0"V17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V17/strided_slicel
V17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V17/Reshape/shape/1
V17/Reshape/shapePackV17/strided_slice:output:0V17/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V17/Reshape/shape
V17/ReshapeReshapefeatures_v17V17/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V17/ReshapeR
	V18/ShapeShapefeatures_v18*
T0*
_output_shapes
:2
	V18/Shape|
V18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V18/strided_slice/stack
V18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V18/strided_slice/stack_1
V18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V18/strided_slice/stack_2ú
V18/strided_sliceStridedSliceV18/Shape:output:0 V18/strided_slice/stack:output:0"V18/strided_slice/stack_1:output:0"V18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V18/strided_slicel
V18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V18/Reshape/shape/1
V18/Reshape/shapePackV18/strided_slice:output:0V18/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V18/Reshape/shape
V18/ReshapeReshapefeatures_v18V18/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V18/ReshapeR
	V19/ShapeShapefeatures_v19*
T0*
_output_shapes
:2
	V19/Shape|
V19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V19/strided_slice/stack
V19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V19/strided_slice/stack_1
V19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V19/strided_slice/stack_2ú
V19/strided_sliceStridedSliceV19/Shape:output:0 V19/strided_slice/stack:output:0"V19/strided_slice/stack_1:output:0"V19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V19/strided_slicel
V19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V19/Reshape/shape/1
V19/Reshape/shapePackV19/strided_slice:output:0V19/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V19/Reshape/shape
V19/ReshapeReshapefeatures_v19V19/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V19/ReshapeO
V2/ShapeShapefeatures_v2*
T0*
_output_shapes
:2

V2/Shapez
V2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V2/strided_slice/stack~
V2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V2/strided_slice/stack_1~
V2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V2/strided_slice/stack_2ô
V2/strided_sliceStridedSliceV2/Shape:output:0V2/strided_slice/stack:output:0!V2/strided_slice/stack_1:output:0!V2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V2/strided_slicej
V2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V2/Reshape/shape/1
V2/Reshape/shapePackV2/strided_slice:output:0V2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V2/Reshape/shape}

V2/ReshapeReshapefeatures_v2V2/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V2/ReshapeR
	V20/ShapeShapefeatures_v20*
T0*
_output_shapes
:2
	V20/Shape|
V20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V20/strided_slice/stack
V20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V20/strided_slice/stack_1
V20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V20/strided_slice/stack_2ú
V20/strided_sliceStridedSliceV20/Shape:output:0 V20/strided_slice/stack:output:0"V20/strided_slice/stack_1:output:0"V20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V20/strided_slicel
V20/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V20/Reshape/shape/1
V20/Reshape/shapePackV20/strided_slice:output:0V20/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V20/Reshape/shape
V20/ReshapeReshapefeatures_v20V20/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V20/ReshapeR
	V21/ShapeShapefeatures_v21*
T0*
_output_shapes
:2
	V21/Shape|
V21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V21/strided_slice/stack
V21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V21/strided_slice/stack_1
V21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V21/strided_slice/stack_2ú
V21/strided_sliceStridedSliceV21/Shape:output:0 V21/strided_slice/stack:output:0"V21/strided_slice/stack_1:output:0"V21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V21/strided_slicel
V21/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V21/Reshape/shape/1
V21/Reshape/shapePackV21/strided_slice:output:0V21/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V21/Reshape/shape
V21/ReshapeReshapefeatures_v21V21/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V21/ReshapeR
	V22/ShapeShapefeatures_v22*
T0*
_output_shapes
:2
	V22/Shape|
V22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V22/strided_slice/stack
V22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V22/strided_slice/stack_1
V22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V22/strided_slice/stack_2ú
V22/strided_sliceStridedSliceV22/Shape:output:0 V22/strided_slice/stack:output:0"V22/strided_slice/stack_1:output:0"V22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V22/strided_slicel
V22/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V22/Reshape/shape/1
V22/Reshape/shapePackV22/strided_slice:output:0V22/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V22/Reshape/shape
V22/ReshapeReshapefeatures_v22V22/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V22/ReshapeR
	V23/ShapeShapefeatures_v23*
T0*
_output_shapes
:2
	V23/Shape|
V23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V23/strided_slice/stack
V23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V23/strided_slice/stack_1
V23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V23/strided_slice/stack_2ú
V23/strided_sliceStridedSliceV23/Shape:output:0 V23/strided_slice/stack:output:0"V23/strided_slice/stack_1:output:0"V23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V23/strided_slicel
V23/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V23/Reshape/shape/1
V23/Reshape/shapePackV23/strided_slice:output:0V23/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V23/Reshape/shape
V23/ReshapeReshapefeatures_v23V23/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V23/ReshapeR
	V24/ShapeShapefeatures_v24*
T0*
_output_shapes
:2
	V24/Shape|
V24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V24/strided_slice/stack
V24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V24/strided_slice/stack_1
V24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V24/strided_slice/stack_2ú
V24/strided_sliceStridedSliceV24/Shape:output:0 V24/strided_slice/stack:output:0"V24/strided_slice/stack_1:output:0"V24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V24/strided_slicel
V24/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V24/Reshape/shape/1
V24/Reshape/shapePackV24/strided_slice:output:0V24/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V24/Reshape/shape
V24/ReshapeReshapefeatures_v24V24/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V24/ReshapeR
	V25/ShapeShapefeatures_v25*
T0*
_output_shapes
:2
	V25/Shape|
V25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V25/strided_slice/stack
V25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V25/strided_slice/stack_1
V25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V25/strided_slice/stack_2ú
V25/strided_sliceStridedSliceV25/Shape:output:0 V25/strided_slice/stack:output:0"V25/strided_slice/stack_1:output:0"V25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V25/strided_slicel
V25/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V25/Reshape/shape/1
V25/Reshape/shapePackV25/strided_slice:output:0V25/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V25/Reshape/shape
V25/ReshapeReshapefeatures_v25V25/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V25/ReshapeR
	V26/ShapeShapefeatures_v26*
T0*
_output_shapes
:2
	V26/Shape|
V26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V26/strided_slice/stack
V26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V26/strided_slice/stack_1
V26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V26/strided_slice/stack_2ú
V26/strided_sliceStridedSliceV26/Shape:output:0 V26/strided_slice/stack:output:0"V26/strided_slice/stack_1:output:0"V26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V26/strided_slicel
V26/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V26/Reshape/shape/1
V26/Reshape/shapePackV26/strided_slice:output:0V26/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V26/Reshape/shape
V26/ReshapeReshapefeatures_v26V26/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V26/ReshapeR
	V27/ShapeShapefeatures_v27*
T0*
_output_shapes
:2
	V27/Shape|
V27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V27/strided_slice/stack
V27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V27/strided_slice/stack_1
V27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V27/strided_slice/stack_2ú
V27/strided_sliceStridedSliceV27/Shape:output:0 V27/strided_slice/stack:output:0"V27/strided_slice/stack_1:output:0"V27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V27/strided_slicel
V27/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V27/Reshape/shape/1
V27/Reshape/shapePackV27/strided_slice:output:0V27/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V27/Reshape/shape
V27/ReshapeReshapefeatures_v27V27/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V27/ReshapeR
	V28/ShapeShapefeatures_v28*
T0*
_output_shapes
:2
	V28/Shape|
V28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V28/strided_slice/stack
V28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V28/strided_slice/stack_1
V28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V28/strided_slice/stack_2ú
V28/strided_sliceStridedSliceV28/Shape:output:0 V28/strided_slice/stack:output:0"V28/strided_slice/stack_1:output:0"V28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V28/strided_slicel
V28/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V28/Reshape/shape/1
V28/Reshape/shapePackV28/strided_slice:output:0V28/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V28/Reshape/shape
V28/ReshapeReshapefeatures_v28V28/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
V28/ReshapeO
V3/ShapeShapefeatures_v3*
T0*
_output_shapes
:2

V3/Shapez
V3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V3/strided_slice/stack~
V3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V3/strided_slice/stack_1~
V3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V3/strided_slice/stack_2ô
V3/strided_sliceStridedSliceV3/Shape:output:0V3/strided_slice/stack:output:0!V3/strided_slice/stack_1:output:0!V3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V3/strided_slicej
V3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V3/Reshape/shape/1
V3/Reshape/shapePackV3/strided_slice:output:0V3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V3/Reshape/shape}

V3/ReshapeReshapefeatures_v3V3/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V3/ReshapeO
V4/ShapeShapefeatures_v4*
T0*
_output_shapes
:2

V4/Shapez
V4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V4/strided_slice/stack~
V4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V4/strided_slice/stack_1~
V4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V4/strided_slice/stack_2ô
V4/strided_sliceStridedSliceV4/Shape:output:0V4/strided_slice/stack:output:0!V4/strided_slice/stack_1:output:0!V4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V4/strided_slicej
V4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V4/Reshape/shape/1
V4/Reshape/shapePackV4/strided_slice:output:0V4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V4/Reshape/shape}

V4/ReshapeReshapefeatures_v4V4/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V4/ReshapeO
V5/ShapeShapefeatures_v5*
T0*
_output_shapes
:2

V5/Shapez
V5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V5/strided_slice/stack~
V5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V5/strided_slice/stack_1~
V5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V5/strided_slice/stack_2ô
V5/strided_sliceStridedSliceV5/Shape:output:0V5/strided_slice/stack:output:0!V5/strided_slice/stack_1:output:0!V5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V5/strided_slicej
V5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V5/Reshape/shape/1
V5/Reshape/shapePackV5/strided_slice:output:0V5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V5/Reshape/shape}

V5/ReshapeReshapefeatures_v5V5/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V5/ReshapeO
V6/ShapeShapefeatures_v6*
T0*
_output_shapes
:2

V6/Shapez
V6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V6/strided_slice/stack~
V6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V6/strided_slice/stack_1~
V6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V6/strided_slice/stack_2ô
V6/strided_sliceStridedSliceV6/Shape:output:0V6/strided_slice/stack:output:0!V6/strided_slice/stack_1:output:0!V6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V6/strided_slicej
V6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V6/Reshape/shape/1
V6/Reshape/shapePackV6/strided_slice:output:0V6/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V6/Reshape/shape}

V6/ReshapeReshapefeatures_v6V6/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V6/ReshapeO
V7/ShapeShapefeatures_v7*
T0*
_output_shapes
:2

V7/Shapez
V7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V7/strided_slice/stack~
V7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V7/strided_slice/stack_1~
V7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V7/strided_slice/stack_2ô
V7/strided_sliceStridedSliceV7/Shape:output:0V7/strided_slice/stack:output:0!V7/strided_slice/stack_1:output:0!V7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V7/strided_slicej
V7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V7/Reshape/shape/1
V7/Reshape/shapePackV7/strided_slice:output:0V7/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V7/Reshape/shape}

V7/ReshapeReshapefeatures_v7V7/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V7/ReshapeO
V8/ShapeShapefeatures_v8*
T0*
_output_shapes
:2

V8/Shapez
V8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V8/strided_slice/stack~
V8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V8/strided_slice/stack_1~
V8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V8/strided_slice/stack_2ô
V8/strided_sliceStridedSliceV8/Shape:output:0V8/strided_slice/stack:output:0!V8/strided_slice/stack_1:output:0!V8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V8/strided_slicej
V8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V8/Reshape/shape/1
V8/Reshape/shapePackV8/strided_slice:output:0V8/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V8/Reshape/shape}

V8/ReshapeReshapefeatures_v8V8/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V8/ReshapeO
V9/ShapeShapefeatures_v9*
T0*
_output_shapes
:2

V9/Shapez
V9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
V9/strided_slice/stack~
V9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
V9/strided_slice/stack_1~
V9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
V9/strided_slice/stack_2ô
V9/strided_sliceStridedSliceV9/Shape:output:0V9/strided_slice/stack:output:0!V9/strided_slice/stack_1:output:0!V9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
V9/strided_slicej
V9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
V9/Reshape/shape/1
V9/Reshape/shapePackV9/strided_slice:output:0V9/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
V9/Reshape/shape}

V9/ReshapeReshapefeatures_v9V9/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

V9/Reshapee
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
concat/axisü
concatConcatV2Amount/Reshape:output:0Time/Reshape:output:0V1/Reshape:output:0V10/Reshape:output:0V11/Reshape:output:0V12/Reshape:output:0V13/Reshape:output:0V14/Reshape:output:0V15/Reshape:output:0V16/Reshape:output:0V17/Reshape:output:0V18/Reshape:output:0V19/Reshape:output:0V2/Reshape:output:0V20/Reshape:output:0V21/Reshape:output:0V22/Reshape:output:0V23/Reshape:output:0V24/Reshape:output:0V25/Reshape:output:0V26/Reshape:output:0V27/Reshape:output:0V28/Reshape:output:0V3/Reshape:output:0V4/Reshape:output:0V5/Reshape:output:0V6/Reshape:output:0V7/Reshape:output:0V8/Reshape:output:0V9/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ï
_input_shapes½
º:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:X T
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_namefeatures/Amount:VR
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namefeatures/Time:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V1:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V10:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V11:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V12:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V13:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V14:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V15:U	Q
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V16:U
Q
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V17:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V18:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V19:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V2:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V20:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V21:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V22:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V23:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V24:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V25:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V26:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V27:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V28:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V3:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V4:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V5:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V6:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V7:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V8:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V9
Æ*
Ï
G__inference_functional_1_layer_call_and_return_conditional_losses_82405

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
v9
batch_normalization_82368
batch_normalization_82370
batch_normalization_82372
batch_normalization_82374
dense_82399
dense_82401
identity¢+batch_normalization/StatefulPartitionedCall¢dense/StatefulPartitionedCall
dense_features/PartitionedCallPartitionedCallamounttimev1v10v11v12v13v14v15v16v17v18v19v2v20v21v22v23v24v25v26v27v28v3v4v5v6v7v8v9*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_819962 
dense_features/PartitionedCall¡
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0batch_normalization_82368batch_normalization_82370batch_normalization_82372batch_normalization_82374*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_816442-
+batch_normalization/StatefulPartitionedCall°
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_82399dense_82401*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_823882
dense/StatefulPartitionedCallÈ
IdentityIdentity&dense/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*ç
_input_shapesÕ
Ò:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameAmount:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameTime:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV1:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV10:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV11:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV12:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV13:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV14:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV15:L	H
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV16:L
H
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV17:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV18:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV19:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV2:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV20:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV21:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV22:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV23:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV24:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV25:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV26:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV27:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV28:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV3:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV4:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV5:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV6:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV7:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV8:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV9
­
¨
@__inference_dense_layer_call_and_return_conditional_losses_84111

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
¦
3__inference_batch_normalization_layer_call_fn_84100

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_816772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
.
ü
G__inference_functional_1_layer_call_and_return_conditional_losses_82627

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
	inputs_29
batch_normalization_82612
batch_normalization_82614
batch_normalization_82616
batch_normalization_82618
dense_82621
dense_82623
identity¢+batch_normalization/StatefulPartitionedCall¢dense/StatefulPartitionedCall¸
dense_features/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16	inputs_17	inputs_18	inputs_19	inputs_20	inputs_21	inputs_22	inputs_23	inputs_24	inputs_25	inputs_26	inputs_27	inputs_28	inputs_29*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_822712 
dense_features/PartitionedCall£
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0batch_normalization_82612batch_normalization_82614batch_normalization_82616batch_normalization_82618*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_816772-
+batch_normalization/StatefulPartitionedCall°
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_82621dense_82623*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_823882
dense/StatefulPartitionedCallÈ
IdentityIdentity&dense/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*ç
_input_shapesÕ
Ò:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:O	K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:O
K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯
¦
3__inference_batch_normalization_layer_call_fn_84087

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_816442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â$
Õ
.__inference_dense_features_layer_call_fn_84018
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
identityû
PartitionedCallPartitionedCallfeatures_amountfeatures_timefeatures_v1features_v10features_v11features_v12features_v13features_v14features_v15features_v16features_v17features_v18features_v19features_v2features_v20features_v21features_v22features_v23features_v24features_v25features_v26features_v27features_v28features_v3features_v4features_v5features_v6features_v7features_v8features_v9*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_822712
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ï
_input_shapes½
º:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:X T
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_namefeatures/Amount:VR
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namefeatures/Time:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V1:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V10:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V11:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V12:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V13:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V14:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V15:U	Q
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V16:U
Q
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V17:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V18:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V19:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V2:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V20:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V21:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V22:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V23:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V24:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V25:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V26:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V27:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namefeatures/V28:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V3:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V4:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V5:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V6:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V7:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V8:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefeatures/V9
)
Å
N__inference_batch_normalization_layer_call_and_return_conditional_losses_84054

inputs
assignmovingavg_84029
assignmovingavg_1_84035)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*(
_class
loc:@AssignMovingAvg/84029*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_84029*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpÂ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*(
_class
loc:@AssignMovingAvg/84029*
_output_shapes
:2
AssignMovingAvg/sub¹
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*(
_class
loc:@AssignMovingAvg/84029*
_output_shapes
:2
AssignMovingAvg/mulÿ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_84029AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/84029*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp£
AssignMovingAvg_1/decayConst**
_class 
loc:@AssignMovingAvg_1/84035*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_84035*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÌ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/84035*
_output_shapes
:2
AssignMovingAvg_1/subÃ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/84035*
_output_shapes
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_84035AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/84035*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1µ
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È*
Ï
G__inference_functional_1_layer_call_and_return_conditional_losses_82453

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
v9
batch_normalization_82438
batch_normalization_82440
batch_normalization_82442
batch_normalization_82444
dense_82447
dense_82449
identity¢+batch_normalization/StatefulPartitionedCall¢dense/StatefulPartitionedCall
dense_features/PartitionedCallPartitionedCallamounttimev1v10v11v12v13v14v15v16v17v18v19v2v20v21v22v23v24v25v26v27v28v3v4v5v6v7v8v9*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_822712 
dense_features/PartitionedCall£
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0batch_normalization_82438batch_normalization_82440batch_normalization_82442batch_normalization_82444*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_816772-
+batch_normalization/StatefulPartitionedCall°
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_82447dense_82449*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_823882
dense/StatefulPartitionedCallÈ
IdentityIdentity&dense/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*ç
_input_shapesÕ
Ò:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameAmount:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameTime:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV1:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV10:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV11:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV12:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV13:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV14:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV15:L	H
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV16:L
H
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV17:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV18:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV19:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV2:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV20:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV21:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV22:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV23:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV24:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV25:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV26:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV27:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV28:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV3:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV4:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV5:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV6:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV7:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV8:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV9
ü 
º
,__inference_functional_1_layer_call_fn_82548

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
v9
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÏ
StatefulPartitionedCallStatefulPartitionedCallamounttimev1v10v11v12v13v14v15v16v17v18v19v2v20v21v22v23v24v25v26v27v28v3v4v5v6v7v8v9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
 !"#*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_825332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*ç
_input_shapesÕ
Ò:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameAmount:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameTime:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV1:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV10:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV11:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV12:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV13:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV14:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV15:L	H
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV16:L
H
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV17:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV18:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV19:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV2:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV20:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV21:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV22:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV23:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV24:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV25:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV26:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV27:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV28:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV3:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV4:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV5:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV6:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV7:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV8:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV9
Î 
±
#__inference_signature_wrapper_82696

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
v9
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallamounttimev1v10v11v12v13v14v15v16v17v18v19v2v20v21v22v23v24v25v26v27v28v3v4v5v6v7v8v9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

 !"#*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_815482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*ç
_input_shapesÕ
Ò:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameAmount:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameTime:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV1:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV10:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV11:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV12:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV13:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV14:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV15:L	H
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV16:L
H
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV17:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV18:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV19:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV2:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV20:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV21:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV22:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV23:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV24:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV25:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV26:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV27:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV28:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV3:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV4:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV5:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV6:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV7:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV8:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameV9
î^
Ô
!__inference__traced_restore_84314
file_prefix.
*assignvariableop_batch_normalization_gamma/
+assignvariableop_1_batch_normalization_beta6
2assignvariableop_2_batch_normalization_moving_mean:
6assignvariableop_3_batch_normalization_moving_variance#
assignvariableop_4_dense_kernel!
assignvariableop_5_dense_bias
assignvariableop_6_sgd_iter 
assignvariableop_7_sgd_decay(
$assignvariableop_8_sgd_learning_rate#
assignvariableop_9_sgd_momentum
assignvariableop_10_total
assignvariableop_11_count
assignvariableop_12_total_1
assignvariableop_13_count_1&
"assignvariableop_14_true_positives&
"assignvariableop_15_true_negatives'
#assignvariableop_16_false_positives'
#assignvariableop_17_false_negatives>
:assignvariableop_18_sgd_batch_normalization_gamma_momentum=
9assignvariableop_19_sgd_batch_normalization_beta_momentum1
-assignvariableop_20_sgd_dense_kernel_momentum/
+assignvariableop_21_sgd_dense_bias_momentum
identity_23¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9÷
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueù
Bö
B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¼
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity©
AssignVariableOpAssignVariableOp*assignvariableop_batch_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1°
AssignVariableOp_1AssignVariableOp+assignvariableop_1_batch_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2·
AssignVariableOp_2AssignVariableOp2assignvariableop_2_batch_normalization_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3»
AssignVariableOp_3AssignVariableOp6assignvariableop_3_batch_normalization_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¤
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¢
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6 
AssignVariableOp_6AssignVariableOpassignvariableop_6_sgd_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¡
AssignVariableOp_7AssignVariableOpassignvariableop_7_sgd_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8©
AssignVariableOp_8AssignVariableOp$assignvariableop_8_sgd_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¤
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_momentumIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¡
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¡
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12£
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13£
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ª
AssignVariableOp_14AssignVariableOp"assignvariableop_14_true_positivesIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ª
AssignVariableOp_15AssignVariableOp"assignvariableop_15_true_negativesIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16«
AssignVariableOp_16AssignVariableOp#assignvariableop_16_false_positivesIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17«
AssignVariableOp_17AssignVariableOp#assignvariableop_17_false_negativesIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Â
AssignVariableOp_18AssignVariableOp:assignvariableop_18_sgd_batch_normalization_gamma_momentumIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Á
AssignVariableOp_19AssignVariableOp9assignvariableop_19_sgd_batch_normalization_beta_momentumIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20µ
AssignVariableOp_20AssignVariableOp-assignvariableop_20_sgd_dense_kernel_momentumIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21³
AssignVariableOp_21AssignVariableOp+assignvariableop_21_sgd_dense_bias_momentumIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_219
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÂ
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_22µ
Identity_23IdentityIdentity_22:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_23"#
identity_23Identity_23:output:0*m
_input_shapes\
Z: ::::::::::::::::::::::2$
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
_user_specified_namefile_prefix


N__inference_batch_normalization_layer_call_and_return_conditional_losses_81677

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
9
Amount/
serving_default_Amount:0ÿÿÿÿÿÿÿÿÿ
5
Time-
serving_default_Time:0ÿÿÿÿÿÿÿÿÿ
1
V1+
serving_default_V1:0ÿÿÿÿÿÿÿÿÿ
3
V10,
serving_default_V10:0ÿÿÿÿÿÿÿÿÿ
3
V11,
serving_default_V11:0ÿÿÿÿÿÿÿÿÿ
3
V12,
serving_default_V12:0ÿÿÿÿÿÿÿÿÿ
3
V13,
serving_default_V13:0ÿÿÿÿÿÿÿÿÿ
3
V14,
serving_default_V14:0ÿÿÿÿÿÿÿÿÿ
3
V15,
serving_default_V15:0ÿÿÿÿÿÿÿÿÿ
3
V16,
serving_default_V16:0ÿÿÿÿÿÿÿÿÿ
3
V17,
serving_default_V17:0ÿÿÿÿÿÿÿÿÿ
3
V18,
serving_default_V18:0ÿÿÿÿÿÿÿÿÿ
3
V19,
serving_default_V19:0ÿÿÿÿÿÿÿÿÿ
1
V2+
serving_default_V2:0ÿÿÿÿÿÿÿÿÿ
3
V20,
serving_default_V20:0ÿÿÿÿÿÿÿÿÿ
3
V21,
serving_default_V21:0ÿÿÿÿÿÿÿÿÿ
3
V22,
serving_default_V22:0ÿÿÿÿÿÿÿÿÿ
3
V23,
serving_default_V23:0ÿÿÿÿÿÿÿÿÿ
3
V24,
serving_default_V24:0ÿÿÿÿÿÿÿÿÿ
3
V25,
serving_default_V25:0ÿÿÿÿÿÿÿÿÿ
3
V26,
serving_default_V26:0ÿÿÿÿÿÿÿÿÿ
3
V27,
serving_default_V27:0ÿÿÿÿÿÿÿÿÿ
3
V28,
serving_default_V28:0ÿÿÿÿÿÿÿÿÿ
1
V3+
serving_default_V3:0ÿÿÿÿÿÿÿÿÿ
1
V4+
serving_default_V4:0ÿÿÿÿÿÿÿÿÿ
1
V5+
serving_default_V5:0ÿÿÿÿÿÿÿÿÿ
1
V6+
serving_default_V6:0ÿÿÿÿÿÿÿÿÿ
1
V7+
serving_default_V7:0ÿÿÿÿÿÿÿÿÿ
1
V8+
serving_default_V8:0ÿÿÿÿÿÿÿÿÿ
1
V9+
serving_default_V9:0ÿÿÿÿÿÿÿÿÿ9
dense0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:þÆ
ª
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
$regularization_losses
%trainable_variables
&	keras_api
'
signatures
*k&call_and_return_all_conditional_losses
l__call__
m_default_save_signature"æ
_tf_keras_networkÉ{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Amount"}, "name": "Amount", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Time"}, "name": "Time", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V1"}, "name": "V1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V10"}, "name": "V10", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V11"}, "name": "V11", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V12"}, "name": "V12", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V13"}, "name": "V13", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V14"}, "name": "V14", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V15"}, "name": "V15", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V16"}, "name": "V16", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V17"}, "name": "V17", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V18"}, "name": "V18", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V19"}, "name": "V19", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V2"}, "name": "V2", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V20"}, "name": "V20", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V21"}, "name": "V21", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V22"}, "name": "V22", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V23"}, "name": "V23", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V24"}, "name": "V24", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V25"}, "name": "V25", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V26"}, "name": "V26", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V27"}, "name": "V27", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V28"}, "name": "V28", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V3"}, "name": "V3", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V4"}, "name": "V4", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V5"}, "name": "V5", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V6"}, "name": "V6", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V7"}, "name": "V7", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V8"}, "name": "V8", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V9"}, "name": "V9", "inbound_nodes": []}, {"class_name": "DenseFeatures", "config": {"name": "dense_features", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "NumericColumn", "config": {"key": "Amount", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "Time", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V1", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V10", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V11", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V12", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V13", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V14", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V15", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V16", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V17", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V18", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V19", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V2", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V20", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V21", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V22", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V23", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V24", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V25", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V26", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V27", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V28", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V3", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V4", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V5", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V6", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V7", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V8", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V9", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}], "partitioner": null}, "name": "dense_features", "inbound_nodes": [{"Time": ["Time", 0, 0, {}], "V1": ["V1", 0, 0, {}], "V2": ["V2", 0, 0, {}], "V3": ["V3", 0, 0, {}], "V4": ["V4", 0, 0, {}], "V5": ["V5", 0, 0, {}], "V6": ["V6", 0, 0, {}], "V7": ["V7", 0, 0, {}], "V8": ["V8", 0, 0, {}], "V9": ["V9", 0, 0, {}], "V10": ["V10", 0, 0, {}], "V11": ["V11", 0, 0, {}], "V12": ["V12", 0, 0, {}], "V13": ["V13", 0, 0, {}], "V14": ["V14", 0, 0, {}], "V15": ["V15", 0, 0, {}], "V16": ["V16", 0, 0, {}], "V17": ["V17", 0, 0, {}], "V18": ["V18", 0, 0, {}], "V19": ["V19", 0, 0, {}], "V20": ["V20", 0, 0, {}], "V21": ["V21", 0, 0, {}], "V22": ["V22", 0, 0, {}], "V23": ["V23", 0, 0, {}], "V24": ["V24", 0, 0, {}], "V25": ["V25", 0, 0, {}], "V26": ["V26", 0, 0, {}], "V27": ["V27", 0, 0, {}], "V28": ["V28", 0, 0, {}], "Amount": ["Amount", 0, 0, {}]}]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["dense_features", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}], "input_layers": {"Time": ["Time", 0, 0], "V1": ["V1", 0, 0], "V2": ["V2", 0, 0], "V3": ["V3", 0, 0], "V4": ["V4", 0, 0], "V5": ["V5", 0, 0], "V6": ["V6", 0, 0], "V7": ["V7", 0, 0], "V8": ["V8", 0, 0], "V9": ["V9", 0, 0], "V10": ["V10", 0, 0], "V11": ["V11", 0, 0], "V12": ["V12", 0, 0], "V13": ["V13", 0, 0], "V14": ["V14", 0, 0], "V15": ["V15", 0, 0], "V16": ["V16", 0, 0], "V17": ["V17", 0, 0], "V18": ["V18", 0, 0], "V19": ["V19", 0, 0], "V20": ["V20", 0, 0], "V21": ["V21", 0, 0], "V22": ["V22", 0, 0], "V23": ["V23", 0, 0], "V24": ["V24", 0, 0], "V25": ["V25", 0, 0], "V26": ["V26", 0, 0], "V27": ["V27", 0, 0], "V28": ["V28", 0, 0], "Amount": ["Amount", 0, 0]}, "output_layers": [["dense", 0, 0]]}, "build_input_shape": {"Time": {"class_name": "TensorShape", "items": [null, 1]}, "V1": {"class_name": "TensorShape", "items": [null, 1]}, "V2": {"class_name": "TensorShape", "items": [null, 1]}, "V3": {"class_name": "TensorShape", "items": [null, 1]}, "V4": {"class_name": "TensorShape", "items": [null, 1]}, "V5": {"class_name": "TensorShape", "items": [null, 1]}, "V6": {"class_name": "TensorShape", "items": [null, 1]}, "V7": {"class_name": "TensorShape", "items": [null, 1]}, "V8": {"class_name": "TensorShape", "items": [null, 1]}, "V9": {"class_name": "TensorShape", "items": [null, 1]}, "V10": {"class_name": "TensorShape", "items": [null, 1]}, "V11": {"class_name": "TensorShape", "items": [null, 1]}, "V12": {"class_name": "TensorShape", "items": [null, 1]}, "V13": {"class_name": "TensorShape", "items": [null, 1]}, "V14": {"class_name": "TensorShape", "items": [null, 1]}, "V15": {"class_name": "TensorShape", "items": [null, 1]}, "V16": {"class_name": "TensorShape", "items": [null, 1]}, "V17": {"class_name": "TensorShape", "items": [null, 1]}, "V18": {"class_name": "TensorShape", "items": [null, 1]}, "V19": {"class_name": "TensorShape", "items": [null, 1]}, "V20": {"class_name": "TensorShape", "items": [null, 1]}, "V21": {"class_name": "TensorShape", "items": [null, 1]}, "V22": {"class_name": "TensorShape", "items": [null, 1]}, "V23": {"class_name": "TensorShape", "items": [null, 1]}, "V24": {"class_name": "TensorShape", "items": [null, 1]}, "V25": {"class_name": "TensorShape", "items": [null, 1]}, "V26": {"class_name": "TensorShape", "items": [null, 1]}, "V27": {"class_name": "TensorShape", "items": [null, 1]}, "V28": {"class_name": "TensorShape", "items": [null, 1]}, "Amount": {"class_name": "TensorShape", "items": [null, 1]}}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Amount"}, "name": "Amount", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Time"}, "name": "Time", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V1"}, "name": "V1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V10"}, "name": "V10", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V11"}, "name": "V11", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V12"}, "name": "V12", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V13"}, "name": "V13", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V14"}, "name": "V14", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V15"}, "name": "V15", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V16"}, "name": "V16", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V17"}, "name": "V17", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V18"}, "name": "V18", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V19"}, "name": "V19", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V2"}, "name": "V2", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V20"}, "name": "V20", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V21"}, "name": "V21", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V22"}, "name": "V22", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V23"}, "name": "V23", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V24"}, "name": "V24", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V25"}, "name": "V25", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V26"}, "name": "V26", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V27"}, "name": "V27", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V28"}, "name": "V28", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V3"}, "name": "V3", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V4"}, "name": "V4", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V5"}, "name": "V5", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V6"}, "name": "V6", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V7"}, "name": "V7", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V8"}, "name": "V8", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V9"}, "name": "V9", "inbound_nodes": []}, {"class_name": "DenseFeatures", "config": {"name": "dense_features", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "NumericColumn", "config": {"key": "Amount", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "Time", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V1", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V10", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V11", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V12", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V13", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V14", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V15", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V16", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V17", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V18", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V19", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V2", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V20", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V21", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V22", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V23", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V24", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V25", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V26", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V27", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V28", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V3", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V4", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V5", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V6", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V7", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V8", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V9", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}], "partitioner": null}, "name": "dense_features", "inbound_nodes": [{"Time": ["Time", 0, 0, {}], "V1": ["V1", 0, 0, {}], "V2": ["V2", 0, 0, {}], "V3": ["V3", 0, 0, {}], "V4": ["V4", 0, 0, {}], "V5": ["V5", 0, 0, {}], "V6": ["V6", 0, 0, {}], "V7": ["V7", 0, 0, {}], "V8": ["V8", 0, 0, {}], "V9": ["V9", 0, 0, {}], "V10": ["V10", 0, 0, {}], "V11": ["V11", 0, 0, {}], "V12": ["V12", 0, 0, {}], "V13": ["V13", 0, 0, {}], "V14": ["V14", 0, 0, {}], "V15": ["V15", 0, 0, {}], "V16": ["V16", 0, 0, {}], "V17": ["V17", 0, 0, {}], "V18": ["V18", 0, 0, {}], "V19": ["V19", 0, 0, {}], "V20": ["V20", 0, 0, {}], "V21": ["V21", 0, 0, {}], "V22": ["V22", 0, 0, {}], "V23": ["V23", 0, 0, {}], "V24": ["V24", 0, 0, {}], "V25": ["V25", 0, 0, {}], "V26": ["V26", 0, 0, {}], "V27": ["V27", 0, 0, {}], "V28": ["V28", 0, 0, {}], "Amount": ["Amount", 0, 0, {}]}]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["dense_features", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}], "input_layers": {"Time": ["Time", 0, 0], "V1": ["V1", 0, 0], "V2": ["V2", 0, 0], "V3": ["V3", 0, 0], "V4": ["V4", 0, 0], "V5": ["V5", 0, 0], "V6": ["V6", 0, 0], "V7": ["V7", 0, 0], "V8": ["V8", 0, 0], "V9": ["V9", 0, 0], "V10": ["V10", 0, 0], "V11": ["V11", 0, 0], "V12": ["V12", 0, 0], "V13": ["V13", 0, 0], "V14": ["V14", 0, 0], "V15": ["V15", 0, 0], "V16": ["V16", 0, 0], "V17": ["V17", 0, 0], "V18": ["V18", 0, 0], "V19": ["V19", 0, 0], "V20": ["V20", 0, 0], "V21": ["V21", 0, 0], "V22": ["V22", 0, 0], "V23": ["V23", 0, 0], "V24": ["V24", 0, 0], "V25": ["V25", 0, 0], "V26": ["V26", 0, 0], "V27": ["V27", 0, 0], "V28": ["V28", 0, 0], "Amount": ["Amount", 0, 0]}, "output_layers": [["dense", 0, 0]]}}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": ["accuracy", {"class_name": "AUC", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "PR", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.02146453969180584, "decay": 0.0, "momentum": 0.860851526260376, "nesterov": false}}}}
ç"ä
_tf_keras_input_layerÄ{"class_name": "InputLayer", "name": "Amount", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Amount"}}
ã"à
_tf_keras_input_layerÀ{"class_name": "InputLayer", "name": "Time", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Time"}}
ß"Ü
_tf_keras_input_layer¼{"class_name": "InputLayer", "name": "V1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V1"}}
á"Þ
_tf_keras_input_layer¾{"class_name": "InputLayer", "name": "V10", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V10"}}
á"Þ
_tf_keras_input_layer¾{"class_name": "InputLayer", "name": "V11", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V11"}}
á"Þ
_tf_keras_input_layer¾{"class_name": "InputLayer", "name": "V12", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V12"}}
á"Þ
_tf_keras_input_layer¾{"class_name": "InputLayer", "name": "V13", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V13"}}
á"Þ
_tf_keras_input_layer¾{"class_name": "InputLayer", "name": "V14", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V14"}}
á"Þ
_tf_keras_input_layer¾{"class_name": "InputLayer", "name": "V15", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V15"}}
á"Þ
_tf_keras_input_layer¾{"class_name": "InputLayer", "name": "V16", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V16"}}
á"Þ
_tf_keras_input_layer¾{"class_name": "InputLayer", "name": "V17", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V17"}}
á"Þ
_tf_keras_input_layer¾{"class_name": "InputLayer", "name": "V18", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V18"}}
á"Þ
_tf_keras_input_layer¾{"class_name": "InputLayer", "name": "V19", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V19"}}
ß"Ü
_tf_keras_input_layer¼{"class_name": "InputLayer", "name": "V2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V2"}}
á"Þ
_tf_keras_input_layer¾{"class_name": "InputLayer", "name": "V20", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V20"}}
á"Þ
_tf_keras_input_layer¾{"class_name": "InputLayer", "name": "V21", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V21"}}
á"Þ
_tf_keras_input_layer¾{"class_name": "InputLayer", "name": "V22", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V22"}}
á"Þ
_tf_keras_input_layer¾{"class_name": "InputLayer", "name": "V23", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V23"}}
á"Þ
_tf_keras_input_layer¾{"class_name": "InputLayer", "name": "V24", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V24"}}
á"Þ
_tf_keras_input_layer¾{"class_name": "InputLayer", "name": "V25", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V25"}}
á"Þ
_tf_keras_input_layer¾{"class_name": "InputLayer", "name": "V26", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V26"}}
á"Þ
_tf_keras_input_layer¾{"class_name": "InputLayer", "name": "V27", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V27"}}
á"Þ
_tf_keras_input_layer¾{"class_name": "InputLayer", "name": "V28", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V28"}}
ß"Ü
_tf_keras_input_layer¼{"class_name": "InputLayer", "name": "V3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V3"}}
ß"Ü
_tf_keras_input_layer¼{"class_name": "InputLayer", "name": "V4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V4"}}
ß"Ü
_tf_keras_input_layer¼{"class_name": "InputLayer", "name": "V5", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V5"}}
ß"Ü
_tf_keras_input_layer¼{"class_name": "InputLayer", "name": "V6", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V6"}}
ß"Ü
_tf_keras_input_layer¼{"class_name": "InputLayer", "name": "V7", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V7"}}
ß"Ü
_tf_keras_input_layer¼{"class_name": "InputLayer", "name": "V8", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V8"}}
ß"Ü
_tf_keras_input_layer¼{"class_name": "InputLayer", "name": "V9", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "V9"}}
Ç;
(_feature_columns
)
_resources
*	variables
+regularization_losses
,trainable_variables
-	keras_api
*n&call_and_return_all_conditional_losses
o__call__":
_tf_keras_layerø9{"class_name": "DenseFeatures", "name": "dense_features", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_features", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "NumericColumn", "config": {"key": "Amount", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "Time", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V1", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V10", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V11", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V12", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V13", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V14", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V15", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V16", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V17", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V18", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V19", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V2", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V20", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V21", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V22", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V23", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V24", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V25", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V26", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V27", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V28", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V3", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V4", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V5", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V6", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V7", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V8", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "V9", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}], "partitioner": null}, "build_input_shape": {"Time": {"class_name": "TensorShape", "items": [null, 1]}, "V1": {"class_name": "TensorShape", "items": [null, 1]}, "V2": {"class_name": "TensorShape", "items": [null, 1]}, "V3": {"class_name": "TensorShape", "items": [null, 1]}, "V4": {"class_name": "TensorShape", "items": [null, 1]}, "V5": {"class_name": "TensorShape", "items": [null, 1]}, "V6": {"class_name": "TensorShape", "items": [null, 1]}, "V7": {"class_name": "TensorShape", "items": [null, 1]}, "V8": {"class_name": "TensorShape", "items": [null, 1]}, "V9": {"class_name": "TensorShape", "items": [null, 1]}, "V10": {"class_name": "TensorShape", "items": [null, 1]}, "V11": {"class_name": "TensorShape", "items": [null, 1]}, "V12": {"class_name": "TensorShape", "items": [null, 1]}, "V13": {"class_name": "TensorShape", "items": [null, 1]}, "V14": {"class_name": "TensorShape", "items": [null, 1]}, "V15": {"class_name": "TensorShape", "items": [null, 1]}, "V16": {"class_name": "TensorShape", "items": [null, 1]}, "V17": {"class_name": "TensorShape", "items": [null, 1]}, "V18": {"class_name": "TensorShape", "items": [null, 1]}, "V19": {"class_name": "TensorShape", "items": [null, 1]}, "V20": {"class_name": "TensorShape", "items": [null, 1]}, "V21": {"class_name": "TensorShape", "items": [null, 1]}, "V22": {"class_name": "TensorShape", "items": [null, 1]}, "V23": {"class_name": "TensorShape", "items": [null, 1]}, "V24": {"class_name": "TensorShape", "items": [null, 1]}, "V25": {"class_name": "TensorShape", "items": [null, 1]}, "V26": {"class_name": "TensorShape", "items": [null, 1]}, "V27": {"class_name": "TensorShape", "items": [null, 1]}, "V28": {"class_name": "TensorShape", "items": [null, 1]}, "Amount": {"class_name": "TensorShape", "items": [null, 1]}}, "_is_feature_layer": true}
®	
.axis
	/gamma
0beta
1moving_mean
2moving_variance
3	variables
4regularization_losses
5trainable_variables
6	keras_api
*p&call_and_return_all_conditional_losses
q__call__"Ú
_tf_keras_layerÀ{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
î

7kernel
8bias
9	variables
:regularization_losses
;trainable_variables
<	keras_api
*r&call_and_return_all_conditional_losses
s__call__"É
_tf_keras_layer¯{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}

=iter
	>decay
?learning_rate
@momentum/momentumg0momentumh7momentumi8momentumj"
	optimizer
J
/0
01
12
23
74
85"
trackable_list_wrapper
 "
trackable_list_wrapper
<
/0
01
72
83"
trackable_list_wrapper
Ê

Alayers
Bnon_trainable_variables
Clayer_regularization_losses
#	variables
$regularization_losses
Dlayer_metrics
Emetrics
%trainable_variables
l__call__
m_default_save_signature
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
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
­

Flayers
Gmetrics
Hnon_trainable_variables
Ilayer_regularization_losses
*	variables
+regularization_losses
Jlayer_metrics
,trainable_variables
o__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
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
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
­

Klayers
Lmetrics
Mnon_trainable_variables
Nlayer_regularization_losses
3	variables
4regularization_losses
Olayer_metrics
5trainable_variables
q__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
:2dense/kernel
:2
dense/bias
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
­

Players
Qmetrics
Rnon_trainable_variables
Slayer_regularization_losses
9	variables
:regularization_losses
Tlayer_metrics
;trainable_variables
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum

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
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
U0
V1
W2"
trackable_list_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
10
21"
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
»
	Xtotal
	Ycount
Z	variables
[	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ÿ
	\total
	]count
^
_fn_kwargs
_	variables
`	keras_api"¸
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
®"
atrue_positives
btrue_negatives
cfalse_positives
dfalse_negatives
e	variables
f	keras_api"»!
_tf_keras_metric !{"class_name": "AUC", "name": "auc", "dtype": "float32", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "PR", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}
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
:È (2true_positives
:È (2true_negatives
 :È (2false_positives
 :È (2false_negatives
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
ê2ç
G__inference_functional_1_layer_call_and_return_conditional_losses_83010
G__inference_functional_1_layer_call_and_return_conditional_losses_82405
G__inference_functional_1_layer_call_and_return_conditional_losses_83308
G__inference_functional_1_layer_call_and_return_conditional_losses_82453À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
þ2û
,__inference_functional_1_layer_call_fn_82548
,__inference_functional_1_layer_call_fn_83400
,__inference_functional_1_layer_call_fn_82642
,__inference_functional_1_layer_call_fn_83354À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
®
2«

 __inference__wrapped_model_81548

²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *õ¢ñ
îªê
*
Amount 
Amountÿÿÿÿÿÿÿÿÿ
&
Time
Timeÿÿÿÿÿÿÿÿÿ
"
V1
V1ÿÿÿÿÿÿÿÿÿ
$
V10
V10ÿÿÿÿÿÿÿÿÿ
$
V11
V11ÿÿÿÿÿÿÿÿÿ
$
V12
V12ÿÿÿÿÿÿÿÿÿ
$
V13
V13ÿÿÿÿÿÿÿÿÿ
$
V14
V14ÿÿÿÿÿÿÿÿÿ
$
V15
V15ÿÿÿÿÿÿÿÿÿ
$
V16
V16ÿÿÿÿÿÿÿÿÿ
$
V17
V17ÿÿÿÿÿÿÿÿÿ
$
V18
V18ÿÿÿÿÿÿÿÿÿ
$
V19
V19ÿÿÿÿÿÿÿÿÿ
"
V2
V2ÿÿÿÿÿÿÿÿÿ
$
V20
V20ÿÿÿÿÿÿÿÿÿ
$
V21
V21ÿÿÿÿÿÿÿÿÿ
$
V22
V22ÿÿÿÿÿÿÿÿÿ
$
V23
V23ÿÿÿÿÿÿÿÿÿ
$
V24
V24ÿÿÿÿÿÿÿÿÿ
$
V25
V25ÿÿÿÿÿÿÿÿÿ
$
V26
V26ÿÿÿÿÿÿÿÿÿ
$
V27
V27ÿÿÿÿÿÿÿÿÿ
$
V28
V28ÿÿÿÿÿÿÿÿÿ
"
V3
V3ÿÿÿÿÿÿÿÿÿ
"
V4
V4ÿÿÿÿÿÿÿÿÿ
"
V5
V5ÿÿÿÿÿÿÿÿÿ
"
V6
V6ÿÿÿÿÿÿÿÿÿ
"
V7
V7ÿÿÿÿÿÿÿÿÿ
"
V8
V8ÿÿÿÿÿÿÿÿÿ
"
V9
V9ÿÿÿÿÿÿÿÿÿ
ð2í
I__inference_dense_features_layer_call_and_return_conditional_losses_83950
I__inference_dense_features_layer_call_and_return_conditional_losses_83675Ô
Ë²Ç
FullArgSpecE
args=:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
º2·
.__inference_dense_features_layer_call_fn_83984
.__inference_dense_features_layer_call_fn_84018Ô
Ë²Ç
FullArgSpecE
args=:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
N__inference_batch_normalization_layer_call_and_return_conditional_losses_84074
N__inference_batch_normalization_layer_call_and_return_conditional_losses_84054´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¤2¡
3__inference_batch_normalization_layer_call_fn_84100
3__inference_batch_normalization_layer_call_fn_84087´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
@__inference_dense_layer_call_and_return_conditional_losses_84111¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_dense_layer_call_fn_84120¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¹B¶
#__inference_signature_wrapper_82696AmountTimeV1V10V11V12V13V14V15V16V17V18V19V2V20V21V22V23V24V25V26V27V28V3V4V5V6V7V8V9à	
 __inference__wrapped_model_81548»	2/1078	¢ý
õ¢ñ
îªê
*
Amount 
Amountÿÿÿÿÿÿÿÿÿ
&
Time
Timeÿÿÿÿÿÿÿÿÿ
"
V1
V1ÿÿÿÿÿÿÿÿÿ
$
V10
V10ÿÿÿÿÿÿÿÿÿ
$
V11
V11ÿÿÿÿÿÿÿÿÿ
$
V12
V12ÿÿÿÿÿÿÿÿÿ
$
V13
V13ÿÿÿÿÿÿÿÿÿ
$
V14
V14ÿÿÿÿÿÿÿÿÿ
$
V15
V15ÿÿÿÿÿÿÿÿÿ
$
V16
V16ÿÿÿÿÿÿÿÿÿ
$
V17
V17ÿÿÿÿÿÿÿÿÿ
$
V18
V18ÿÿÿÿÿÿÿÿÿ
$
V19
V19ÿÿÿÿÿÿÿÿÿ
"
V2
V2ÿÿÿÿÿÿÿÿÿ
$
V20
V20ÿÿÿÿÿÿÿÿÿ
$
V21
V21ÿÿÿÿÿÿÿÿÿ
$
V22
V22ÿÿÿÿÿÿÿÿÿ
$
V23
V23ÿÿÿÿÿÿÿÿÿ
$
V24
V24ÿÿÿÿÿÿÿÿÿ
$
V25
V25ÿÿÿÿÿÿÿÿÿ
$
V26
V26ÿÿÿÿÿÿÿÿÿ
$
V27
V27ÿÿÿÿÿÿÿÿÿ
$
V28
V28ÿÿÿÿÿÿÿÿÿ
"
V3
V3ÿÿÿÿÿÿÿÿÿ
"
V4
V4ÿÿÿÿÿÿÿÿÿ
"
V5
V5ÿÿÿÿÿÿÿÿÿ
"
V6
V6ÿÿÿÿÿÿÿÿÿ
"
V7
V7ÿÿÿÿÿÿÿÿÿ
"
V8
V8ÿÿÿÿÿÿÿÿÿ
"
V9
V9ÿÿÿÿÿÿÿÿÿ
ª "-ª*
(
dense
denseÿÿÿÿÿÿÿÿÿ´
N__inference_batch_normalization_layer_call_and_return_conditional_losses_84054b12/03¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ´
N__inference_batch_normalization_layer_call_and_return_conditional_losses_84074b2/103¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
3__inference_batch_normalization_layer_call_fn_84087U12/03¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
3__inference_batch_normalization_layer_call_fn_84100U2/103¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
I__inference_dense_features_layer_call_and_return_conditional_losses_83675Á¢
¢
ü
ªø

3
Amount)&
features/Amountÿÿÿÿÿÿÿÿÿ
/
Time'$
features/Timeÿÿÿÿÿÿÿÿÿ
+
V1%"
features/V1ÿÿÿÿÿÿÿÿÿ
-
V10&#
features/V10ÿÿÿÿÿÿÿÿÿ
-
V11&#
features/V11ÿÿÿÿÿÿÿÿÿ
-
V12&#
features/V12ÿÿÿÿÿÿÿÿÿ
-
V13&#
features/V13ÿÿÿÿÿÿÿÿÿ
-
V14&#
features/V14ÿÿÿÿÿÿÿÿÿ
-
V15&#
features/V15ÿÿÿÿÿÿÿÿÿ
-
V16&#
features/V16ÿÿÿÿÿÿÿÿÿ
-
V17&#
features/V17ÿÿÿÿÿÿÿÿÿ
-
V18&#
features/V18ÿÿÿÿÿÿÿÿÿ
-
V19&#
features/V19ÿÿÿÿÿÿÿÿÿ
+
V2%"
features/V2ÿÿÿÿÿÿÿÿÿ
-
V20&#
features/V20ÿÿÿÿÿÿÿÿÿ
-
V21&#
features/V21ÿÿÿÿÿÿÿÿÿ
-
V22&#
features/V22ÿÿÿÿÿÿÿÿÿ
-
V23&#
features/V23ÿÿÿÿÿÿÿÿÿ
-
V24&#
features/V24ÿÿÿÿÿÿÿÿÿ
-
V25&#
features/V25ÿÿÿÿÿÿÿÿÿ
-
V26&#
features/V26ÿÿÿÿÿÿÿÿÿ
-
V27&#
features/V27ÿÿÿÿÿÿÿÿÿ
-
V28&#
features/V28ÿÿÿÿÿÿÿÿÿ
+
V3%"
features/V3ÿÿÿÿÿÿÿÿÿ
+
V4%"
features/V4ÿÿÿÿÿÿÿÿÿ
+
V5%"
features/V5ÿÿÿÿÿÿÿÿÿ
+
V6%"
features/V6ÿÿÿÿÿÿÿÿÿ
+
V7%"
features/V7ÿÿÿÿÿÿÿÿÿ
+
V8%"
features/V8ÿÿÿÿÿÿÿÿÿ
+
V9%"
features/V9ÿÿÿÿÿÿÿÿÿ

 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
I__inference_dense_features_layer_call_and_return_conditional_losses_83950Á¢
¢
ü
ªø

3
Amount)&
features/Amountÿÿÿÿÿÿÿÿÿ
/
Time'$
features/Timeÿÿÿÿÿÿÿÿÿ
+
V1%"
features/V1ÿÿÿÿÿÿÿÿÿ
-
V10&#
features/V10ÿÿÿÿÿÿÿÿÿ
-
V11&#
features/V11ÿÿÿÿÿÿÿÿÿ
-
V12&#
features/V12ÿÿÿÿÿÿÿÿÿ
-
V13&#
features/V13ÿÿÿÿÿÿÿÿÿ
-
V14&#
features/V14ÿÿÿÿÿÿÿÿÿ
-
V15&#
features/V15ÿÿÿÿÿÿÿÿÿ
-
V16&#
features/V16ÿÿÿÿÿÿÿÿÿ
-
V17&#
features/V17ÿÿÿÿÿÿÿÿÿ
-
V18&#
features/V18ÿÿÿÿÿÿÿÿÿ
-
V19&#
features/V19ÿÿÿÿÿÿÿÿÿ
+
V2%"
features/V2ÿÿÿÿÿÿÿÿÿ
-
V20&#
features/V20ÿÿÿÿÿÿÿÿÿ
-
V21&#
features/V21ÿÿÿÿÿÿÿÿÿ
-
V22&#
features/V22ÿÿÿÿÿÿÿÿÿ
-
V23&#
features/V23ÿÿÿÿÿÿÿÿÿ
-
V24&#
features/V24ÿÿÿÿÿÿÿÿÿ
-
V25&#
features/V25ÿÿÿÿÿÿÿÿÿ
-
V26&#
features/V26ÿÿÿÿÿÿÿÿÿ
-
V27&#
features/V27ÿÿÿÿÿÿÿÿÿ
-
V28&#
features/V28ÿÿÿÿÿÿÿÿÿ
+
V3%"
features/V3ÿÿÿÿÿÿÿÿÿ
+
V4%"
features/V4ÿÿÿÿÿÿÿÿÿ
+
V5%"
features/V5ÿÿÿÿÿÿÿÿÿ
+
V6%"
features/V6ÿÿÿÿÿÿÿÿÿ
+
V7%"
features/V7ÿÿÿÿÿÿÿÿÿ
+
V8%"
features/V8ÿÿÿÿÿÿÿÿÿ
+
V9%"
features/V9ÿÿÿÿÿÿÿÿÿ

 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ç
.__inference_dense_features_layer_call_fn_83984´¢
¢
ü
ªø

3
Amount)&
features/Amountÿÿÿÿÿÿÿÿÿ
/
Time'$
features/Timeÿÿÿÿÿÿÿÿÿ
+
V1%"
features/V1ÿÿÿÿÿÿÿÿÿ
-
V10&#
features/V10ÿÿÿÿÿÿÿÿÿ
-
V11&#
features/V11ÿÿÿÿÿÿÿÿÿ
-
V12&#
features/V12ÿÿÿÿÿÿÿÿÿ
-
V13&#
features/V13ÿÿÿÿÿÿÿÿÿ
-
V14&#
features/V14ÿÿÿÿÿÿÿÿÿ
-
V15&#
features/V15ÿÿÿÿÿÿÿÿÿ
-
V16&#
features/V16ÿÿÿÿÿÿÿÿÿ
-
V17&#
features/V17ÿÿÿÿÿÿÿÿÿ
-
V18&#
features/V18ÿÿÿÿÿÿÿÿÿ
-
V19&#
features/V19ÿÿÿÿÿÿÿÿÿ
+
V2%"
features/V2ÿÿÿÿÿÿÿÿÿ
-
V20&#
features/V20ÿÿÿÿÿÿÿÿÿ
-
V21&#
features/V21ÿÿÿÿÿÿÿÿÿ
-
V22&#
features/V22ÿÿÿÿÿÿÿÿÿ
-
V23&#
features/V23ÿÿÿÿÿÿÿÿÿ
-
V24&#
features/V24ÿÿÿÿÿÿÿÿÿ
-
V25&#
features/V25ÿÿÿÿÿÿÿÿÿ
-
V26&#
features/V26ÿÿÿÿÿÿÿÿÿ
-
V27&#
features/V27ÿÿÿÿÿÿÿÿÿ
-
V28&#
features/V28ÿÿÿÿÿÿÿÿÿ
+
V3%"
features/V3ÿÿÿÿÿÿÿÿÿ
+
V4%"
features/V4ÿÿÿÿÿÿÿÿÿ
+
V5%"
features/V5ÿÿÿÿÿÿÿÿÿ
+
V6%"
features/V6ÿÿÿÿÿÿÿÿÿ
+
V7%"
features/V7ÿÿÿÿÿÿÿÿÿ
+
V8%"
features/V8ÿÿÿÿÿÿÿÿÿ
+
V9%"
features/V9ÿÿÿÿÿÿÿÿÿ

 
p
ª "ÿÿÿÿÿÿÿÿÿç
.__inference_dense_features_layer_call_fn_84018´¢
¢
ü
ªø

3
Amount)&
features/Amountÿÿÿÿÿÿÿÿÿ
/
Time'$
features/Timeÿÿÿÿÿÿÿÿÿ
+
V1%"
features/V1ÿÿÿÿÿÿÿÿÿ
-
V10&#
features/V10ÿÿÿÿÿÿÿÿÿ
-
V11&#
features/V11ÿÿÿÿÿÿÿÿÿ
-
V12&#
features/V12ÿÿÿÿÿÿÿÿÿ
-
V13&#
features/V13ÿÿÿÿÿÿÿÿÿ
-
V14&#
features/V14ÿÿÿÿÿÿÿÿÿ
-
V15&#
features/V15ÿÿÿÿÿÿÿÿÿ
-
V16&#
features/V16ÿÿÿÿÿÿÿÿÿ
-
V17&#
features/V17ÿÿÿÿÿÿÿÿÿ
-
V18&#
features/V18ÿÿÿÿÿÿÿÿÿ
-
V19&#
features/V19ÿÿÿÿÿÿÿÿÿ
+
V2%"
features/V2ÿÿÿÿÿÿÿÿÿ
-
V20&#
features/V20ÿÿÿÿÿÿÿÿÿ
-
V21&#
features/V21ÿÿÿÿÿÿÿÿÿ
-
V22&#
features/V22ÿÿÿÿÿÿÿÿÿ
-
V23&#
features/V23ÿÿÿÿÿÿÿÿÿ
-
V24&#
features/V24ÿÿÿÿÿÿÿÿÿ
-
V25&#
features/V25ÿÿÿÿÿÿÿÿÿ
-
V26&#
features/V26ÿÿÿÿÿÿÿÿÿ
-
V27&#
features/V27ÿÿÿÿÿÿÿÿÿ
-
V28&#
features/V28ÿÿÿÿÿÿÿÿÿ
+
V3%"
features/V3ÿÿÿÿÿÿÿÿÿ
+
V4%"
features/V4ÿÿÿÿÿÿÿÿÿ
+
V5%"
features/V5ÿÿÿÿÿÿÿÿÿ
+
V6%"
features/V6ÿÿÿÿÿÿÿÿÿ
+
V7%"
features/V7ÿÿÿÿÿÿÿÿÿ
+
V8%"
features/V8ÿÿÿÿÿÿÿÿÿ
+
V9%"
features/V9ÿÿÿÿÿÿÿÿÿ

 
p 
ª "ÿÿÿÿÿÿÿÿÿ 
@__inference_dense_layer_call_and_return_conditional_losses_84111\78/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 x
%__inference_dense_layer_call_fn_84120O78/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ

G__inference_functional_1_layer_call_and_return_conditional_losses_82405»	12/078	¢	
ý¢ù
îªê
*
Amount 
Amountÿÿÿÿÿÿÿÿÿ
&
Time
Timeÿÿÿÿÿÿÿÿÿ
"
V1
V1ÿÿÿÿÿÿÿÿÿ
$
V10
V10ÿÿÿÿÿÿÿÿÿ
$
V11
V11ÿÿÿÿÿÿÿÿÿ
$
V12
V12ÿÿÿÿÿÿÿÿÿ
$
V13
V13ÿÿÿÿÿÿÿÿÿ
$
V14
V14ÿÿÿÿÿÿÿÿÿ
$
V15
V15ÿÿÿÿÿÿÿÿÿ
$
V16
V16ÿÿÿÿÿÿÿÿÿ
$
V17
V17ÿÿÿÿÿÿÿÿÿ
$
V18
V18ÿÿÿÿÿÿÿÿÿ
$
V19
V19ÿÿÿÿÿÿÿÿÿ
"
V2
V2ÿÿÿÿÿÿÿÿÿ
$
V20
V20ÿÿÿÿÿÿÿÿÿ
$
V21
V21ÿÿÿÿÿÿÿÿÿ
$
V22
V22ÿÿÿÿÿÿÿÿÿ
$
V23
V23ÿÿÿÿÿÿÿÿÿ
$
V24
V24ÿÿÿÿÿÿÿÿÿ
$
V25
V25ÿÿÿÿÿÿÿÿÿ
$
V26
V26ÿÿÿÿÿÿÿÿÿ
$
V27
V27ÿÿÿÿÿÿÿÿÿ
$
V28
V28ÿÿÿÿÿÿÿÿÿ
"
V3
V3ÿÿÿÿÿÿÿÿÿ
"
V4
V4ÿÿÿÿÿÿÿÿÿ
"
V5
V5ÿÿÿÿÿÿÿÿÿ
"
V6
V6ÿÿÿÿÿÿÿÿÿ
"
V7
V7ÿÿÿÿÿÿÿÿÿ
"
V8
V8ÿÿÿÿÿÿÿÿÿ
"
V9
V9ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 

G__inference_functional_1_layer_call_and_return_conditional_losses_82453»	2/1078	¢	
ý¢ù
îªê
*
Amount 
Amountÿÿÿÿÿÿÿÿÿ
&
Time
Timeÿÿÿÿÿÿÿÿÿ
"
V1
V1ÿÿÿÿÿÿÿÿÿ
$
V10
V10ÿÿÿÿÿÿÿÿÿ
$
V11
V11ÿÿÿÿÿÿÿÿÿ
$
V12
V12ÿÿÿÿÿÿÿÿÿ
$
V13
V13ÿÿÿÿÿÿÿÿÿ
$
V14
V14ÿÿÿÿÿÿÿÿÿ
$
V15
V15ÿÿÿÿÿÿÿÿÿ
$
V16
V16ÿÿÿÿÿÿÿÿÿ
$
V17
V17ÿÿÿÿÿÿÿÿÿ
$
V18
V18ÿÿÿÿÿÿÿÿÿ
$
V19
V19ÿÿÿÿÿÿÿÿÿ
"
V2
V2ÿÿÿÿÿÿÿÿÿ
$
V20
V20ÿÿÿÿÿÿÿÿÿ
$
V21
V21ÿÿÿÿÿÿÿÿÿ
$
V22
V22ÿÿÿÿÿÿÿÿÿ
$
V23
V23ÿÿÿÿÿÿÿÿÿ
$
V24
V24ÿÿÿÿÿÿÿÿÿ
$
V25
V25ÿÿÿÿÿÿÿÿÿ
$
V26
V26ÿÿÿÿÿÿÿÿÿ
$
V27
V27ÿÿÿÿÿÿÿÿÿ
$
V28
V28ÿÿÿÿÿÿÿÿÿ
"
V3
V3ÿÿÿÿÿÿÿÿÿ
"
V4
V4ÿÿÿÿÿÿÿÿÿ
"
V5
V5ÿÿÿÿÿÿÿÿÿ
"
V6
V6ÿÿÿÿÿÿÿÿÿ
"
V7
V7ÿÿÿÿÿÿÿÿÿ
"
V8
V8ÿÿÿÿÿÿÿÿÿ
"
V9
V9ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ù
G__inference_functional_1_layer_call_and_return_conditional_losses_8301012/078Û
¢×

Ï
¢Ë

À
ª¼

1
Amount'$
inputs/Amountÿÿÿÿÿÿÿÿÿ
-
Time%"
inputs/Timeÿÿÿÿÿÿÿÿÿ
)
V1# 
	inputs/V1ÿÿÿÿÿÿÿÿÿ
+
V10$!

inputs/V10ÿÿÿÿÿÿÿÿÿ
+
V11$!

inputs/V11ÿÿÿÿÿÿÿÿÿ
+
V12$!

inputs/V12ÿÿÿÿÿÿÿÿÿ
+
V13$!

inputs/V13ÿÿÿÿÿÿÿÿÿ
+
V14$!

inputs/V14ÿÿÿÿÿÿÿÿÿ
+
V15$!

inputs/V15ÿÿÿÿÿÿÿÿÿ
+
V16$!

inputs/V16ÿÿÿÿÿÿÿÿÿ
+
V17$!

inputs/V17ÿÿÿÿÿÿÿÿÿ
+
V18$!

inputs/V18ÿÿÿÿÿÿÿÿÿ
+
V19$!

inputs/V19ÿÿÿÿÿÿÿÿÿ
)
V2# 
	inputs/V2ÿÿÿÿÿÿÿÿÿ
+
V20$!

inputs/V20ÿÿÿÿÿÿÿÿÿ
+
V21$!

inputs/V21ÿÿÿÿÿÿÿÿÿ
+
V22$!

inputs/V22ÿÿÿÿÿÿÿÿÿ
+
V23$!

inputs/V23ÿÿÿÿÿÿÿÿÿ
+
V24$!

inputs/V24ÿÿÿÿÿÿÿÿÿ
+
V25$!

inputs/V25ÿÿÿÿÿÿÿÿÿ
+
V26$!

inputs/V26ÿÿÿÿÿÿÿÿÿ
+
V27$!

inputs/V27ÿÿÿÿÿÿÿÿÿ
+
V28$!

inputs/V28ÿÿÿÿÿÿÿÿÿ
)
V3# 
	inputs/V3ÿÿÿÿÿÿÿÿÿ
)
V4# 
	inputs/V4ÿÿÿÿÿÿÿÿÿ
)
V5# 
	inputs/V5ÿÿÿÿÿÿÿÿÿ
)
V6# 
	inputs/V6ÿÿÿÿÿÿÿÿÿ
)
V7# 
	inputs/V7ÿÿÿÿÿÿÿÿÿ
)
V8# 
	inputs/V8ÿÿÿÿÿÿÿÿÿ
)
V9# 
	inputs/V9ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ù
G__inference_functional_1_layer_call_and_return_conditional_losses_833082/1078Û
¢×

Ï
¢Ë

À
ª¼

1
Amount'$
inputs/Amountÿÿÿÿÿÿÿÿÿ
-
Time%"
inputs/Timeÿÿÿÿÿÿÿÿÿ
)
V1# 
	inputs/V1ÿÿÿÿÿÿÿÿÿ
+
V10$!

inputs/V10ÿÿÿÿÿÿÿÿÿ
+
V11$!

inputs/V11ÿÿÿÿÿÿÿÿÿ
+
V12$!

inputs/V12ÿÿÿÿÿÿÿÿÿ
+
V13$!

inputs/V13ÿÿÿÿÿÿÿÿÿ
+
V14$!

inputs/V14ÿÿÿÿÿÿÿÿÿ
+
V15$!

inputs/V15ÿÿÿÿÿÿÿÿÿ
+
V16$!

inputs/V16ÿÿÿÿÿÿÿÿÿ
+
V17$!

inputs/V17ÿÿÿÿÿÿÿÿÿ
+
V18$!

inputs/V18ÿÿÿÿÿÿÿÿÿ
+
V19$!

inputs/V19ÿÿÿÿÿÿÿÿÿ
)
V2# 
	inputs/V2ÿÿÿÿÿÿÿÿÿ
+
V20$!

inputs/V20ÿÿÿÿÿÿÿÿÿ
+
V21$!

inputs/V21ÿÿÿÿÿÿÿÿÿ
+
V22$!

inputs/V22ÿÿÿÿÿÿÿÿÿ
+
V23$!

inputs/V23ÿÿÿÿÿÿÿÿÿ
+
V24$!

inputs/V24ÿÿÿÿÿÿÿÿÿ
+
V25$!

inputs/V25ÿÿÿÿÿÿÿÿÿ
+
V26$!

inputs/V26ÿÿÿÿÿÿÿÿÿ
+
V27$!

inputs/V27ÿÿÿÿÿÿÿÿÿ
+
V28$!

inputs/V28ÿÿÿÿÿÿÿÿÿ
)
V3# 
	inputs/V3ÿÿÿÿÿÿÿÿÿ
)
V4# 
	inputs/V4ÿÿÿÿÿÿÿÿÿ
)
V5# 
	inputs/V5ÿÿÿÿÿÿÿÿÿ
)
V6# 
	inputs/V6ÿÿÿÿÿÿÿÿÿ
)
V7# 
	inputs/V7ÿÿÿÿÿÿÿÿÿ
)
V8# 
	inputs/V8ÿÿÿÿÿÿÿÿÿ
)
V9# 
	inputs/V9ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ß	
,__inference_functional_1_layer_call_fn_82548®	12/078	¢	
ý¢ù
îªê
*
Amount 
Amountÿÿÿÿÿÿÿÿÿ
&
Time
Timeÿÿÿÿÿÿÿÿÿ
"
V1
V1ÿÿÿÿÿÿÿÿÿ
$
V10
V10ÿÿÿÿÿÿÿÿÿ
$
V11
V11ÿÿÿÿÿÿÿÿÿ
$
V12
V12ÿÿÿÿÿÿÿÿÿ
$
V13
V13ÿÿÿÿÿÿÿÿÿ
$
V14
V14ÿÿÿÿÿÿÿÿÿ
$
V15
V15ÿÿÿÿÿÿÿÿÿ
$
V16
V16ÿÿÿÿÿÿÿÿÿ
$
V17
V17ÿÿÿÿÿÿÿÿÿ
$
V18
V18ÿÿÿÿÿÿÿÿÿ
$
V19
V19ÿÿÿÿÿÿÿÿÿ
"
V2
V2ÿÿÿÿÿÿÿÿÿ
$
V20
V20ÿÿÿÿÿÿÿÿÿ
$
V21
V21ÿÿÿÿÿÿÿÿÿ
$
V22
V22ÿÿÿÿÿÿÿÿÿ
$
V23
V23ÿÿÿÿÿÿÿÿÿ
$
V24
V24ÿÿÿÿÿÿÿÿÿ
$
V25
V25ÿÿÿÿÿÿÿÿÿ
$
V26
V26ÿÿÿÿÿÿÿÿÿ
$
V27
V27ÿÿÿÿÿÿÿÿÿ
$
V28
V28ÿÿÿÿÿÿÿÿÿ
"
V3
V3ÿÿÿÿÿÿÿÿÿ
"
V4
V4ÿÿÿÿÿÿÿÿÿ
"
V5
V5ÿÿÿÿÿÿÿÿÿ
"
V6
V6ÿÿÿÿÿÿÿÿÿ
"
V7
V7ÿÿÿÿÿÿÿÿÿ
"
V8
V8ÿÿÿÿÿÿÿÿÿ
"
V9
V9ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿß	
,__inference_functional_1_layer_call_fn_82642®	2/1078	¢	
ý¢ù
îªê
*
Amount 
Amountÿÿÿÿÿÿÿÿÿ
&
Time
Timeÿÿÿÿÿÿÿÿÿ
"
V1
V1ÿÿÿÿÿÿÿÿÿ
$
V10
V10ÿÿÿÿÿÿÿÿÿ
$
V11
V11ÿÿÿÿÿÿÿÿÿ
$
V12
V12ÿÿÿÿÿÿÿÿÿ
$
V13
V13ÿÿÿÿÿÿÿÿÿ
$
V14
V14ÿÿÿÿÿÿÿÿÿ
$
V15
V15ÿÿÿÿÿÿÿÿÿ
$
V16
V16ÿÿÿÿÿÿÿÿÿ
$
V17
V17ÿÿÿÿÿÿÿÿÿ
$
V18
V18ÿÿÿÿÿÿÿÿÿ
$
V19
V19ÿÿÿÿÿÿÿÿÿ
"
V2
V2ÿÿÿÿÿÿÿÿÿ
$
V20
V20ÿÿÿÿÿÿÿÿÿ
$
V21
V21ÿÿÿÿÿÿÿÿÿ
$
V22
V22ÿÿÿÿÿÿÿÿÿ
$
V23
V23ÿÿÿÿÿÿÿÿÿ
$
V24
V24ÿÿÿÿÿÿÿÿÿ
$
V25
V25ÿÿÿÿÿÿÿÿÿ
$
V26
V26ÿÿÿÿÿÿÿÿÿ
$
V27
V27ÿÿÿÿÿÿÿÿÿ
$
V28
V28ÿÿÿÿÿÿÿÿÿ
"
V3
V3ÿÿÿÿÿÿÿÿÿ
"
V4
V4ÿÿÿÿÿÿÿÿÿ
"
V5
V5ÿÿÿÿÿÿÿÿÿ
"
V6
V6ÿÿÿÿÿÿÿÿÿ
"
V7
V7ÿÿÿÿÿÿÿÿÿ
"
V8
V8ÿÿÿÿÿÿÿÿÿ
"
V9
V9ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ±
,__inference_functional_1_layer_call_fn_8335412/078Û
¢×

Ï
¢Ë

À
ª¼

1
Amount'$
inputs/Amountÿÿÿÿÿÿÿÿÿ
-
Time%"
inputs/Timeÿÿÿÿÿÿÿÿÿ
)
V1# 
	inputs/V1ÿÿÿÿÿÿÿÿÿ
+
V10$!

inputs/V10ÿÿÿÿÿÿÿÿÿ
+
V11$!

inputs/V11ÿÿÿÿÿÿÿÿÿ
+
V12$!

inputs/V12ÿÿÿÿÿÿÿÿÿ
+
V13$!

inputs/V13ÿÿÿÿÿÿÿÿÿ
+
V14$!

inputs/V14ÿÿÿÿÿÿÿÿÿ
+
V15$!

inputs/V15ÿÿÿÿÿÿÿÿÿ
+
V16$!

inputs/V16ÿÿÿÿÿÿÿÿÿ
+
V17$!

inputs/V17ÿÿÿÿÿÿÿÿÿ
+
V18$!

inputs/V18ÿÿÿÿÿÿÿÿÿ
+
V19$!

inputs/V19ÿÿÿÿÿÿÿÿÿ
)
V2# 
	inputs/V2ÿÿÿÿÿÿÿÿÿ
+
V20$!

inputs/V20ÿÿÿÿÿÿÿÿÿ
+
V21$!

inputs/V21ÿÿÿÿÿÿÿÿÿ
+
V22$!

inputs/V22ÿÿÿÿÿÿÿÿÿ
+
V23$!

inputs/V23ÿÿÿÿÿÿÿÿÿ
+
V24$!

inputs/V24ÿÿÿÿÿÿÿÿÿ
+
V25$!

inputs/V25ÿÿÿÿÿÿÿÿÿ
+
V26$!

inputs/V26ÿÿÿÿÿÿÿÿÿ
+
V27$!

inputs/V27ÿÿÿÿÿÿÿÿÿ
+
V28$!

inputs/V28ÿÿÿÿÿÿÿÿÿ
)
V3# 
	inputs/V3ÿÿÿÿÿÿÿÿÿ
)
V4# 
	inputs/V4ÿÿÿÿÿÿÿÿÿ
)
V5# 
	inputs/V5ÿÿÿÿÿÿÿÿÿ
)
V6# 
	inputs/V6ÿÿÿÿÿÿÿÿÿ
)
V7# 
	inputs/V7ÿÿÿÿÿÿÿÿÿ
)
V8# 
	inputs/V8ÿÿÿÿÿÿÿÿÿ
)
V9# 
	inputs/V9ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ±
,__inference_functional_1_layer_call_fn_834002/1078Û
¢×

Ï
¢Ë

À
ª¼

1
Amount'$
inputs/Amountÿÿÿÿÿÿÿÿÿ
-
Time%"
inputs/Timeÿÿÿÿÿÿÿÿÿ
)
V1# 
	inputs/V1ÿÿÿÿÿÿÿÿÿ
+
V10$!

inputs/V10ÿÿÿÿÿÿÿÿÿ
+
V11$!

inputs/V11ÿÿÿÿÿÿÿÿÿ
+
V12$!

inputs/V12ÿÿÿÿÿÿÿÿÿ
+
V13$!

inputs/V13ÿÿÿÿÿÿÿÿÿ
+
V14$!

inputs/V14ÿÿÿÿÿÿÿÿÿ
+
V15$!

inputs/V15ÿÿÿÿÿÿÿÿÿ
+
V16$!

inputs/V16ÿÿÿÿÿÿÿÿÿ
+
V17$!

inputs/V17ÿÿÿÿÿÿÿÿÿ
+
V18$!

inputs/V18ÿÿÿÿÿÿÿÿÿ
+
V19$!

inputs/V19ÿÿÿÿÿÿÿÿÿ
)
V2# 
	inputs/V2ÿÿÿÿÿÿÿÿÿ
+
V20$!

inputs/V20ÿÿÿÿÿÿÿÿÿ
+
V21$!

inputs/V21ÿÿÿÿÿÿÿÿÿ
+
V22$!

inputs/V22ÿÿÿÿÿÿÿÿÿ
+
V23$!

inputs/V23ÿÿÿÿÿÿÿÿÿ
+
V24$!

inputs/V24ÿÿÿÿÿÿÿÿÿ
+
V25$!

inputs/V25ÿÿÿÿÿÿÿÿÿ
+
V26$!

inputs/V26ÿÿÿÿÿÿÿÿÿ
+
V27$!

inputs/V27ÿÿÿÿÿÿÿÿÿ
+
V28$!

inputs/V28ÿÿÿÿÿÿÿÿÿ
)
V3# 
	inputs/V3ÿÿÿÿÿÿÿÿÿ
)
V4# 
	inputs/V4ÿÿÿÿÿÿÿÿÿ
)
V5# 
	inputs/V5ÿÿÿÿÿÿÿÿÿ
)
V6# 
	inputs/V6ÿÿÿÿÿÿÿÿÿ
)
V7# 
	inputs/V7ÿÿÿÿÿÿÿÿÿ
)
V8# 
	inputs/V8ÿÿÿÿÿÿÿÿÿ
)
V9# 
	inputs/V9ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÜ	
#__inference_signature_wrapper_82696´	2/1078ú¢ö
¢ 
îªê
*
Amount 
Amountÿÿÿÿÿÿÿÿÿ
&
Time
Timeÿÿÿÿÿÿÿÿÿ
"
V1
V1ÿÿÿÿÿÿÿÿÿ
$
V10
V10ÿÿÿÿÿÿÿÿÿ
$
V11
V11ÿÿÿÿÿÿÿÿÿ
$
V12
V12ÿÿÿÿÿÿÿÿÿ
$
V13
V13ÿÿÿÿÿÿÿÿÿ
$
V14
V14ÿÿÿÿÿÿÿÿÿ
$
V15
V15ÿÿÿÿÿÿÿÿÿ
$
V16
V16ÿÿÿÿÿÿÿÿÿ
$
V17
V17ÿÿÿÿÿÿÿÿÿ
$
V18
V18ÿÿÿÿÿÿÿÿÿ
$
V19
V19ÿÿÿÿÿÿÿÿÿ
"
V2
V2ÿÿÿÿÿÿÿÿÿ
$
V20
V20ÿÿÿÿÿÿÿÿÿ
$
V21
V21ÿÿÿÿÿÿÿÿÿ
$
V22
V22ÿÿÿÿÿÿÿÿÿ
$
V23
V23ÿÿÿÿÿÿÿÿÿ
$
V24
V24ÿÿÿÿÿÿÿÿÿ
$
V25
V25ÿÿÿÿÿÿÿÿÿ
$
V26
V26ÿÿÿÿÿÿÿÿÿ
$
V27
V27ÿÿÿÿÿÿÿÿÿ
$
V28
V28ÿÿÿÿÿÿÿÿÿ
"
V3
V3ÿÿÿÿÿÿÿÿÿ
"
V4
V4ÿÿÿÿÿÿÿÿÿ
"
V5
V5ÿÿÿÿÿÿÿÿÿ
"
V6
V6ÿÿÿÿÿÿÿÿÿ
"
V7
V7ÿÿÿÿÿÿÿÿÿ
"
V8
V8ÿÿÿÿÿÿÿÿÿ
"
V9
V9ÿÿÿÿÿÿÿÿÿ"-ª*
(
dense
denseÿÿÿÿÿÿÿÿÿ