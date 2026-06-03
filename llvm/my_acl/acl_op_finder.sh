: << 'COMMENT'
Все "опасные" случаи, т.е. непонятные для обычного grep-анализа

~/tmp/pytorch/torch_npu/_inductor/npu_device.py:177:                cmd.Name(kernel_name.c_str())
    это кастомный вызыватель функций, загруженных через *.npubin... нужен для triton, а не самого torch_npu
~/tmp/pytorch/torch_npu/csrc/inductor/mlir/cpp_common.cpp:168:    cmd.Name(name).SetCustomHandler(launch_call).Run();
    ~/tmp/pytorch/torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/npu/codegen/cpp_wrapper.py:122:    opcommand_call("{kernel_name}", launch_call);
    ~/tmp/pytorch/torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/npu/codegen/cpp_wrapper.py:150:    opcommand_call("{kernel_name}", launch_call);
    ~/tmp/pytorch/torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/npu/codegen/cpp_wrapper.py:334:  opcommand_call("{kernel_name}", launch_call);
    вообще что-то чисто для python

~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BitwiseNotKernelNpu.cpp:28:    cmd.Name(real_op_name)
    string real_op_name = (self.dtype() == at::kBool) ? "LogicalNot" : "Invert";
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/__Ior__KernelNpu.cpp:28:    cmd.Name(real_op_name).Input(self).Input(other).Output(result).Run();
    string real_op_name = (self.dtype() == at::ScalarType::Bool) ? "LogicalOr" : "BitwiseOr";
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/__Ior__KernelNpu.cpp:36:    cmd.Name(real_op_name).Input(self).Input(other, self.scalar_type()).Output(result).Run();
    string real_op_name = (self.dtype() == at::kBool) ? "LogicalOr" : "BitwiseOr";
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BitwiseAndKernelNpu.cpp:28:    cmd.Name(real_op_name).Input(self).Input(other, self.scalar_type()).Output(result).Run();
    string real_op_name = (self.dtype() == at::kBool) ? "LogicalAnd" : "BitwiseAnd";
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BitwiseAndKernelNpu.cpp:42:        cmd.Name(real_op_name).Expect(unified_result).Input(self).Input(other).Output(result).Run();
    string real_op_name = (self.dtype() == at::kBool) ? "LogicalAnd" : "BitwiseAnd";
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BatchNormBackwardKernelNpu.cpp:43:    cmd.Name(name)
    string name = (self.dim() == 5) ? "BN3DTrainingUpdateGrad" : "BNTrainingUpdateGrad";
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BatchNormBackwardKernelNpu.cpp:81:    cmd.Name(name)
    string name = (self.dim() == 5) ? "BN3DTrainingReduceGrad" : "BNTrainingReduceGrad";
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BatchNormKernelNpu.cpp:66:    cmd.Name(name)
    string name = (self.dim() == 5) ? "BN3DTrainingReduce" : "BNTrainingReduce";
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BatchNormKernelNpu.cpp:93:    cmd.Name(name)
    string name = (self.dim() == 5) ? "BN3DTrainingUpdate" : "BNTrainingUpdate";
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BitwiseOrKernelNpu.cpp:28:    cmd.Name(real_op_name).Input(self).Input(other, self.scalar_type()).Output(result).Run();
    string real_op_name = (self.dtype() == at::kBool) ? "LogicalOr" : "BitwiseOr";
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BitwiseOrKernelNpu.cpp:42:        cmd.Name(real_op_name).Expect(unified_result).Input(self).Input(other).Output(result).Run();
    string real_op_name = (self.dtype() == at::kBool) ? "LogicalOr" : "BitwiseOr";
COMMENT
MISC="LogicalNot Invert LogicalOr BitwiseOr LogicalAnd BitwiseAnd BN3DTrainingUpdateGrad BNTrainingUpdateGrad BN3DTrainingUpdate BNTrainingUpdate BN3DTrainingReduceGrad BNTrainingReduceGrad BN3DTrainingReduce BNTrainingReduce"

ops=$(grep 'cmd.Name' ~/tmp/pytorch -rn \
    | grep -o 'cmd.Name("[^"]*")' \
    | sed -E 's/cmd.Name\("([^"]*)"\)/\1/' \
    | sort -u)
ops=$(printf "%s\n%s\n" "$ops" "$MISC" | tr ' ' '\n' | sort -u)

for op in $ops; do
    echo "    $op"
    src1=$(grep "cmd.Name(\"$op\")" ~/tmp/pytorch -rn)
    src2=$(grep -E "\? \"$op\" : \"[^\"]*\"|\? \"[^\"]*\" : \"$op\"" ~/tmp/pytorch -rn)
    src=$(printf "%s\n%s\n" "$src1" "$src2" | grep -v '^[[:space:]]*$' | sort -u | sed "s|$HOME|~|")
    echo "$src"
done

: << 'COMMENT'
финальный результат поисков (всего 369 операций):

    Acos
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AcosKernelNpu.cpp:27:    cmd.Name("Acos")
    Acosh
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AcoshKernelNpu.cpp:27:    cmd.Name("Acosh")
    AdaptiveAvgPool2d
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AdaptiveAvgPool2dKernelNpu.cpp:52:        cmd.Name("AdaptiveAvgPool2d")
    AdaptiveAvgPool2dGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AdaptiveAvgPool2dBackwardKernelNpu.cpp:50:        cmd.Name("AdaptiveAvgPool2dGrad")
    AddLayerNorm
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AddLayerNormNpu.cpp:44:    cmd.Name("AddLayerNorm")
    AddLayerNormGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AddLayerNormBackwardKernelNpu.cpp:47:    cmd.Name("AddLayerNormGrad")
    AddRmsNorm
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AddRmsNormKernelNpu.cpp:53:    cmd.Name("AddRmsNorm")
    Addcdiv
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AddcdivKernelNpu.cpp:32:    cmd.Name("Addcdiv")
    Addcmul
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AddcmulKernelNpu.cpp:32:    cmd.Name("Addcmul")
    AffineGrid
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AffineGridGeneratorKernelNpu.cpp:31:    cmd.Name("AffineGrid")
    AnchorResponseFlags
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AnchorResponseFlagsKernelNpu.cpp:55:    cmd.Name("AnchorResponseFlags")
    ApplyAdamD
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ApplyAdamKernelNpu.cpp:39:    cmd.Name("ApplyAdamD")
    ApplyAdamV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BertApplyAdamKernelNpu.cpp:39:  cmd.Name("ApplyAdamV2")
    ApplyAdamW
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ApplyAdamWKernelNpu.cpp:40:  cmd.Name("ApplyAdamW")
    ArgMaxGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ScatterV1KernelNpu.cpp:26:    cmd.Name("ArgMaxGrad")
    ArgMaxV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ArgmaxKernelNpu.cpp:28:    cmd.Name("ArgMaxV2")
    ArgMaxWithValue
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MaxKernelNpu.cpp:31:    cmd.Name("ArgMaxWithValue")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MaxV1KernelNpu.cpp:30:  cmd.Name("ArgMaxWithValue")
    ArgMin
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ArgminKernelNpu.cpp:30:    cmd.Name("ArgMin")
    ArgMinWithValue
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MinKernelNpu.cpp:35:    cmd.Name("ArgMinWithValue")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MinV1KernelNpu.cpp:31:    cmd.Name("ArgMinWithValue")
    AsStrided
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AsStridedKernelNpu.cpp:70:        cmd.Name("AsStrided")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AsStridedKernelNpu.cpp:86:        cmd.Name("AsStrided")
    AscendQuantV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/QuantizeKernelNpu.cpp:81:    cmd.Name("AscendQuantV2")
    Asin
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AsinKernelNpu.cpp:26:  cmd.Name("Asin")
    Asinh
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AsinhKernelNpu.cpp:27:    cmd.Name("Asinh")
    Atan
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AtanKernelNpu.cpp:27:    cmd.Name("Atan")
    Atan2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/Atan2KernelNpu.cpp:31:    cmd.Name("Atan2")
    Atanh
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AtanhKernelNpu.cpp:26:  cmd.Name("Atanh")
    AttentionLnQKV
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FusedAttentionLnQKV.cpp:51:    cmd.Name("AttentionLnQKV")
    AttentionScore
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FusedAttentionScoreKernelNpu.cpp:72:  cmd.Name("AttentionScore")
    AttentionScoreGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FusedAttentionScoreKernelNpu.cpp:111:  cmd.Name("AttentionScoreGrad")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FusedAttentionScoreKernelNpu.cpp:208:    cmd.Name("AttentionScoreGrad")
    AvgPool3D
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AvgPool3dKernelNpu.cpp:39:    cmd.Name("AvgPool3D")
    AvgPool3DGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AvgPool3dBackwardKernelNpu.cpp:49:    cmd.Name("AvgPool3DGrad")
    AvgPoolV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AvgPool2dKernelNpu.cpp:96:    cmd.Name("AvgPoolV2")
    AvgPoolV2Grad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AvgPool2dBackwardKernelNpu.cpp:50:    cmd.Name("AvgPoolV2Grad")
    AxpyV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AddKernelNpu.cpp:96:            cmd.Name("AxpyV2").Input(self).Input(other).Input(alpha, self.scalar_type()).Output(result).Run();
    AxpyWithSoftmaxAndDropOutDoMask
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DropoutWithAddSoftmaxKernelNpu.cpp:66:    cmd.Name("AxpyWithSoftmaxAndDropOutDoMask")
    BN3DTrainingReduce
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BatchNormKernelNpu.cpp:65:    string name = (self.dim() == 5) ? "BN3DTrainingReduce" : "BNTrainingReduce";
    BN3DTrainingReduceGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BatchNormBackwardKernelNpu.cpp:72:    string name = (self.dim() == 5) ? "BN3DTrainingReduceGrad" : "BNTrainingReduceGrad";
    BN3DTrainingUpdate
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BatchNormKernelNpu.cpp:92:    string name = (self.dim() == 5) ? "BN3DTrainingUpdate" : "BNTrainingUpdate";
    BN3DTrainingUpdateGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BatchNormBackwardKernelNpu.cpp:42:    string name = (self.dim() == 5) ? "BN3DTrainingUpdateGrad" : "BNTrainingUpdateGrad";
    BNInfer
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BatchNormKernelNpu.cpp:39:    cmd.Name("BNInfer")
    BNInferGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BatchNormBackwardKernelNpu.cpp:113:    cmd.Name("BNInferGrad")
    BNTrainingReduce
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BatchNormKernelNpu.cpp:65:    string name = (self.dim() == 5) ? "BN3DTrainingReduce" : "BNTrainingReduce";
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BatchNormReduceKernelNpu.cpp:35:    cmd.Name("BNTrainingReduce")
    BNTrainingReduceGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BatchNormBackwardKernelNpu.cpp:72:    string name = (self.dim() == 5) ? "BN3DTrainingReduceGrad" : "BNTrainingReduceGrad";
    BNTrainingUpdate
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BatchNormKernelNpu.cpp:92:    string name = (self.dim() == 5) ? "BN3DTrainingUpdate" : "BNTrainingUpdate";
    BNTrainingUpdateGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BatchNormBackwardKernelNpu.cpp:42:    string name = (self.dim() == 5) ? "BN3DTrainingUpdateGrad" : "BNTrainingUpdateGrad";
    BatchMatMul
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AddbmmKernelNpu.cpp:35:    cmd.Name("BatchMatMul")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AffineGridGeneratorBackwardKernelNpu.cpp:52:    cmd.Name("BatchMatMul")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BaddbmmKernelNpu.cpp:40:    cmd.Name("BatchMatMul")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BmmKernelNpu.cpp:37:        cmd.Name("BatchMatMul")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BmmV2KernelNpu.cpp:110:    cmd.Name("BatchMatMul")
    BatchMultiClassNonMaxSuppression
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BatchNMSKernelNpu.cpp:33:    cmd.Name("BatchMultiClassNonMaxSuppression")
    BiasAddGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SlowConvDilated2dBackwardKernelNpu.cpp:92:    cmd.Name("BiasAddGrad").Input(self).Output(grad_bias).Attr("data_format", data_formats).Run();
    BinaryCrossEntropy
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BinaryCrossEntropyKernelNpu.cpp:34:  cmd.Name("BinaryCrossEntropy")
    BinaryCrossEntropyGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BinaryCrossEntropyBackwardKernelNpu.cpp:35:    cmd.Name("BinaryCrossEntropyGrad")
    Bincount
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BincountKernelNpu.cpp:50:    cmd.Name("Bincount").Input(input).Input(at::Scalar(sizes), at::kInt).Input(weight).Output(result).Run();
    BitwiseAnd
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BitwiseAndKernelNpu.cpp:26:    string real_op_name = (self.dtype() == at::kBool) ? "LogicalAnd" : "BitwiseAnd";
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BitwiseAndKernelNpu.cpp:40:        string real_op_name = (self.dtype() == at::kBool) ? "LogicalAnd" : "BitwiseAnd";
    BitwiseOr
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BitwiseOrKernelNpu.cpp:26:    string real_op_name = (self.dtype() == at::kBool) ? "LogicalOr" : "BitwiseOr";
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BitwiseOrKernelNpu.cpp:40:        string real_op_name = (self.dtype() == at::kBool) ? "LogicalOr" : "BitwiseOr";
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/__Ior__KernelNpu.cpp:26:    string real_op_name = (self.dtype() == at::ScalarType::Bool) ? "LogicalOr" : "BitwiseOr";
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/__Ior__KernelNpu.cpp:34:    string real_op_name = (self.dtype() == at::kBool) ? "LogicalOr" : "BitwiseOr";
    BitwiseXor
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BitwiseXorKernelNpu.cpp:27:    cmd.Name("BitwiseXor").Input(self).Input(other, self.scalar_type()).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BitwiseXorKernelNpu.cpp:40:        cmd.Name("BitwiseXor").Expect(unified_result).Input(self).Input(other).Output(result).Run();
    BoundingBoxDecode
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BoundingBoxDecodeKernelNpu.cpp:49:    cmd.Name("BoundingBoxDecode")
    BoundingBoxEncode
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BoundingBoxEncodeKernelNpu.cpp:45:  cmd.Name("BoundingBoxEncode")
    BroadcastTo
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BroadcastKernelNpu.cpp:31:    cmd.Name("BroadcastTo")
    CIoU
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/CiouKernelNpu.cpp:55:  cmd.Name("CIoU")
    CIoUGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/CiouKernelNpu.cpp:80:  cmd.Name("CIoUGrad")
    CTCLossV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/CtcLossKernelNpu.cpp:59:    cmd.Name("CTCLossV2")
    CTCLossV2Grad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/CtcLossBackwardKernelNpu.cpp:47:    cmd.Name("CTCLossV2Grad")
    Cast
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/CastKernelNpu.cpp:31:    cmd.Name("Cast")
    Cdist
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/CdistKernelNpu.cpp:90:    cmd.Name("Cdist").Input(tensor1_broadcast).Input(tensor2_broadcast).Attr("p", p_float).Output(result).Run();
    CdistGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/CdistBackwardKernelNpu.cpp:64:    cmd.Name("CdistGrad")
    CeluV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/CeluKernelNpu.cpp:27:  cmd.Name("CeluV2")
    ClipByValue
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/HardtanhKernelNpu.cpp:31:  cmd.Name("ClipByValue")
    ClipByValueV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ClampKernelNpu.cpp:36:    cmd.Name("ClipByValueV2")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ClampKernelNpu.cpp:88:    cmd.Name("ClipByValueV2")
    Col2im
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/Col2imKernelNpu.cpp:36:    cmd.Name("Col2im")
    ConcatD
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/CatKernelNpu.cpp:63:    cmd.Name("ConcatD");
    ConfusionTransposeD
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ConfusionTransposeKernelNpu.cpp:59:    cmd.Name("ConfusionTransposeD")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ConfusionTransposeKernelNpu.cpp:96:    cmd.Name("ConfusionTransposeD")
    Conv2D
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/Conv2dKernelNpu.cpp:52:    cmd.Name("Conv2D").Input(input, "x").Input(weight, "filter");
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ConvTranspose2dBackwardKernelNpu.cpp:42:    cmd.Name("Conv2D")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ConvtbcKernelNpu.cpp:52:    cmd.Name("Conv2D")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/NnpackSpatialConvolutionKernelNpu.cpp:46:    cmd.Name("Conv2D")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SlowConvDilated2DKernelNpu.cpp:53:    cmd.Name("Conv2D")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SlowConvTranspose2dBackwardKernelNpu.cpp:39:    cmd.Name("Conv2D")
    Conv2DBackpropFilter
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/Conv2dBackwardKernelNpu.cpp:117:    cmd.Name("Conv2DBackpropFilter")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ConvTranspose2dBackwardKernelNpu.cpp:74:    cmd.Name("Conv2DBackpropFilter")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SlowConvDilated2dBackwardKernelNpu.cpp:70:    cmd.Name("Conv2DBackpropFilter")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SlowConvTranspose2dBackwardKernelNpu.cpp:66:    cmd.Name("Conv2DBackpropFilter")
    Conv2DBackpropInput
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/Conv2dBackwardKernelNpu.cpp:68:    cmd.Name("Conv2DBackpropInput")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SlowConvDilated2dBackwardKernelNpu.cpp:42:    cmd.Name("Conv2DBackpropInput")
    Conv2DTranspose
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ConvTranspose2dKernelNpu.cpp:41:  cmd.Name("Conv2DTranspose")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SlowConvTranspose2dKernelNpu.cpp:130:    cmd.Name("Conv2DTranspose").Input(size_vec, at::kInt).Input(self, "x").Input(weight, "filter");
    Conv3D
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/Conv3dKernelNpu.cpp:147:    cmd.Name("Conv3D").Input(input, "x").Input(filter, "filter");
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/Conv3dKernelNpu.cpp:78:    cmd.Name("Conv3D").Input(input, "x").Input(filter, "filter");
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ConvTranspose3dBackwardKernelNpu.cpp:44:    cmd.Name("Conv3D")
    Conv3DBackpropFilter
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/Conv3dBackwardKernelNpu.cpp:71:    cmd.Name("Conv3DBackpropFilter")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ConvTranspose3dBackwardKernelNpu.cpp:78:    cmd.Name("Conv3DBackpropFilter")
    Conv3DBackpropInput
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/Conv3dBackwardKernelNpu.cpp:41:    cmd.Name("Conv3DBackpropInput")
    Conv3DTranspose
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ConvTranspose2dBackwardKernelNpu.cpp:176:    cmd.Name("Conv3DTranspose").Input(sizeVec, at::kInt).Input(input).Input(weight);
    Cos
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/CosKernelNpu.cpp:28:  cmd.Name("Cos")
    Cosh
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/CoshKernelNpu.cpp:27:    cmd.Name("Cosh")
    CropAndResizeV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/CropAndResizeKernelNpu.cpp:36:    cmd.Name("CropAndResizeV2")
    Cross
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LinalgCrossKernelNpu.cpp:36:    cmd.Name("Cross").Input(self).Input(other).Output(result).Attr("dim", real_dim).Run();
    Cummax
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/CummaxKernelNpu.cpp:33:    cmd.Name("Cummax")
    Cummin
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/CumminKernelNpu.cpp:25:    cmd.Name("Cummin").Input(self).Output(values).Output(indices).Attr("axis", dim).Run();
    Cumprod
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/CumprodKernelNpu.cpp:31:  cmd.Name("Cumprod")
    Cumsum
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/CumsumKernelNpu.cpp:35:    cmd.Name("Cumsum").Input(self);
    DIoU
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DiouKernelNpu.cpp:47:    cmd.Name("DIoU")
    DIoUGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DiouKernelNpu.cpp:84:    cmd.Name("DIoUGrad")
    DecodeJpeg
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DecodeJpegKernelNpu.cpp:35:    cmd.Name("DecodeJpeg")
    DeepNorm
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DeepNormKernelNpu.cpp:43:    cmd.Name("DeepNorm")
    DeepNormGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DeepNormBackwardKernelNpu.cpp:36:    cmd.Name("DeepNormGrad")
    DeformableOffsets
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DeformableConv2dKernelNpu.cpp:74:    cmd.Name("DeformableOffsets")
    DeformableOffsetsGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DeformableConv2dKernelNpu.cpp:184:    cmd.Name("DeformableOffsetsGrad")
    DepthwiseConv2D
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ConvDepthWise2dKernelNpu.cpp:45:    cmd.Name("DepthwiseConv2D").Input(self, "x").Input(weight_modify, "filter");
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ConvDepthWise2dKernelNpu.cpp:82:    cmd.Name("DepthwiseConv2D").Input(self, "x").Input(weight_modify, "filter");
    DepthwiseConv2DBackpropFilter
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ConvDepthwise2dBackwardKernelNpu.cpp:67:    cmd.Name("DepthwiseConv2DBackpropFilter")
    DepthwiseConv2DBackpropInput
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ConvDepthwise2dBackwardKernelNpu.cpp:39:    cmd.Name("DepthwiseConv2DBackpropInput")
    Diag
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DiagKernelNpu.cpp:51:    cmd.Name("Diag");
    DiagPart
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DiagKernelNpu.cpp:53:    cmd.Name("DiagPart");
    Dot
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DotKernelNpu.cpp:27:    cmd.Name("Dot")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/VDotKernelNpu.cpp:27:    cmd.Name("Dot")
    DropOutDoMask
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DropoutKernelNpu.cpp:199:  cmd.Name("DropOutDoMask")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DropoutKernelNpu.cpp:37:  cmd.Name("DropOutDoMask")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DropoutKernelNpu.cpp:53:  cmd.Name("DropOutDoMask")
    DropOutDoMaskV3
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DropoutWithByteMaskKernelNpu.cpp:122:    cmd.Name("DropOutDoMaskV3")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DropoutWithByteMaskKernelNpu.cpp:35:    cmd.Name("DropOutDoMaskV3")
    DropOutGenMaskV3
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DropoutWithAddSoftmaxKernelNpu.cpp:37:    cmd.Name("DropOutGenMaskV3")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DropoutWithByteMaskKernelNpu.cpp:62:    cmd.Name("DropOutGenMaskV3")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FusedAttentionScoreKernelNpu.cpp:47:  cmd.Name("DropOutGenMaskV3")
    DropoutWithMulsAndSoftmaxGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DropoutWithAddSoftmaxKernelNpu.cpp:91:    cmd.Name("DropoutWithMulsAndSoftmaxGrad")
    DynamicGRUV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/GruKernelNpu.cpp:72:    cmd.Name("DynamicGRUV2")
    DynamicGRUV2Grad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/GruKernelNpu.cpp:297:    cmd.Name("DynamicGRUV2Grad")
    DynamicRNN
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LstmKernelNpu.cpp:55:    cmd.Name("DynamicRNN").Input(input, "x").Input(weight, "w").Input(bias, "b");
    DynamicRNNGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LstmKernelNpu.cpp:359:    cmd.Name("DynamicRNNGrad")
    DynamicRNNV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LstmCellKernelNpu.cpp:52:    cmd.Name("DynamicRNNV2").Input(input_reshape).Input(w_ih).Input(w_hh);
    DynamicRNNV2Grad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LstmCellKernelNpu.cpp:99:    cmd.Name("DynamicRNNV2Grad")
    Elu
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/EluKernelNpu.cpp:36:    cmd.Name("Elu")
    EluGradV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/EluBackwardKernelNpu.cpp:44:  cmd.Name("EluGradV2")
    EmbeddingBag
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/EmbeddingBagKernelNpu.cpp:73:        cmd.Name("EmbeddingBag").Input(weight).Input(indices).Input(offsets);
    EmbeddingDenseGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/EmbeddingDenseBackwardKernelNpu.cpp:37:    cmd.Name("EmbeddingDenseGrad")
    Erf
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ErfKernelNpu.cpp:26:  cmd.Name("Erf")
    Erfc
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ErfcKernelNpu.cpp:27:    cmd.Name("Erfc")
    Erfinv
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ErfinvKernelNpu.cpp:27:    cmd.Name("Erfinv")
    Exp
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ExpKernelNpu.cpp:27:    cmd.Name("Exp")
    Expm1
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/Expm1KernelNpu.cpp:27:    cmd.Name("Expm1")
    Eye
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/EyeKernelNpu.cpp:28:    cmd.Name("Eye")
    FastGelu
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FastGeluKernelNpu.cpp:45:    cmd.Name("FastGelu")
    FastGeluGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FastGeluKernelNpu.cpp:32:    cmd.Name("FastGeluGrad")
    FillDiagonal
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FillDiagonalKernelNpu.cpp:31:    cmd.Name("FillDiagonal")
    Floor
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FloorKernelNpu.cpp:26:  cmd.Name("Floor")
    FloorDiv
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FloorDivideKernelNpu.cpp:27:    cmd.Name("FloorDiv").Input(self).Input(other, self.scalar_type()).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FloorDivideKernelNpu.cpp:34:    cmd.Name("FloorDiv").Input(self, other.scalar_type()).Input(other).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FloorDivideKernelNpu.cpp:46:        cmd.Name("FloorDiv").Input(self).Input(other).Output(result).Run();
    FloorMod
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RemainderKernelNpu.cpp:33:    cmd.Name("FloorMod")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RemainderKernelNpu.cpp:48:    cmd.Name("FloorMod")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RemainderKernelNpu.cpp:63:    cmd.Name("FloorMod")
    GIoU
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/GiouKernelNpu.cpp:42:    cmd.Name("GIoU")
    GIoUGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/GiouKernelNpu.cpp:61:    cmd.Name("GIoUGrad")
    GLU
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/GluKernelNpu.cpp:28:    cmd.Name("GLU")
    GLUGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/GluBackwardKernelNpu.cpp:41:    cmd.Name("GLUGrad")
    Gather
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/TakeKernelNpu.cpp:31:    cmd.Name("Gather")
    GatherElements
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/GatherKernelNpu.cpp:38:    cmd.Name("GatherElements")
    GatherV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/IndexSelectKernelNpu.cpp:40:  cmd.Name("GatherV2")
~/tmp/pytorch/third_party/op-plugin/op_plugin/utils/custom_functions/aclops/EmbeddingKernelNpu.cpp:34:    cmd.Name("GatherV2")
    GatherV2D
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/EmbeddingRenormKernelNpu.cpp:30:    cmd.Name("GatherV2D")
    Gelu
~/tmp/pytorch/third_party/op-plugin/op_plugin/utils/custom_functions/aclops/GeluKernelNpu.cpp:23:    cmd.Name("Gelu")
    GeluGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/utils/custom_functions/aclops/GeluBackwardKernelNpu.cpp:23:    cmd.Name("GeluGrad")
    GridAssignPositive
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/GridAssignPositiveKernelNpu.cpp:64:    cmd.Name("GridAssignPositive")
    GridSampler2D
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/GridSampler2dKernelNpu.cpp:54:    cmd.Name("GridSampler2D")
    GridSampler2DGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/GridSampler2dBackwardKernelNpu.cpp:58:    cmd.Name("GridSampler2DGrad")
    GridSampler3D
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/GridSampler3dKernelNpu.cpp:28:    cmd.Name("GridSampler3D")
    GridSampler3DGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/utils/custom_functions/aclops/GridSampler3dBackwardKernelNpu.cpp:55:    cmd.Name("GridSampler3DGrad")
    GroupNorm
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/GroupNormKernelNpu.cpp:54:    cmd.Name("GroupNorm")
    GroupNormSwish
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/GroupNormKernelNpu.cpp:91:    cmd.Name("GroupNormSwish")
    HardShrink
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/HardShrinkKernelNpu.cpp:31:    cmd.Name("HardShrink")
    HardShrinkGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/HardShrinkBackwardKernelNpu.cpp:30:    cmd.Name("HardShrinkGrad")
    HardSigmoid
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/HardsigmoidKernelNpu.cpp:30:    cmd.Name("HardSigmoid")
    HardSigmoidGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/HardsigmoidBackwardKernelNpu.cpp:29:    cmd.Name("HardSigmoidGrad")
    HardSwish
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/HardSwishKernelNpu.cpp:28:    cmd.Name("HardSwish")
    HardSwishGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/HardSwishBackwardKernelNpu.cpp:27:    cmd.Name("HardSwishGrad")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/HardSwishBackwardKernelNpu.cpp:40:    cmd.Name("HardSwishGrad")
    HardtanhGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/HardtanhBackwardKernelNpu.cpp:33:    cmd.Name("HardtanhGrad")
    Histogram
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/HistcKernelNpu.cpp:29:    cmd.Name("Histogram")
    IFMR
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/IfmrKernelNpu.cpp:42:  cmd.Name("IFMR")
    Identity
~/tmp/pytorch/torch_npu/csrc/aten/common/CopyKernel.cpp:357:    cmd.Name("Identity")
~/tmp/pytorch/torch_npu/csrc/aten/common/FormatCastKernelNpu.cpp:150:    cmd.Name("Identity")
    Im2col
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/Im2colKernelNpu.cpp:97:    cmd.Name("Im2col")
    ImgToTensor
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ImgToTensorKernelNpu.cpp:28:    cmd.Name("ImgToTensor")
    Index
~/tmp/pytorch/third_party/op-plugin/op_plugin/utils/custom_functions/aclops/IndexKernelNpu.cpp:54:    cmd.Name("Index")
    IndexFillD
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/IndexFillKernelNpu.cpp:122:    cmd.Name("IndexFillD").Input(self).Input(assist_help1).Input(assist_help2).Attr("dim", dim).Output(result).Run();
    IndexPutV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/IndexPutKernelNpu.cpp:115:    cmd.Name("IndexPutV2")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/IndexPutKernelNpu.cpp:174:    cmd.Name("IndexPutV2")
    InplaceIndexAdd
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/IndexAddKernelNpu.cpp:44:    cmd.Name("InplaceIndexAdd")
    Invert
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BitwiseNotKernelNpu.cpp:26:    string real_op_name = (self.dtype() == at::kBool) ? "LogicalNot" : "Invert";
    Iou
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/IouKernelNpu.cpp:48:    cmd.Name("Iou")
    IsClose
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/IscloseKernelNpu.cpp:39:    cmd.Name("IsClose")
    KLDiv
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/KlDivKernelNpu.cpp:38:    cmd.Name("KLDiv")
    KlDivLossGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/KlDivBackwardKernelNpu.cpp:39:    cmd.Name("KlDivLossGrad")
    L1LossGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/L1LossKernelNpu.cpp:51:    cmd.Name("L1LossGrad")
    LayerNorm
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LayerNormKernelNpu.cpp:86:        cmd.Name("LayerNorm")
    LayerNormGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LayerNormBackwardKernelNpu.cpp:39:    cmd.Name("LayerNormGrad")
    LayerNormV3
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LayerNormEvalKernelNpu.cpp:74:    cmd.Name("LayerNormV3")
    LeakyRelu
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LeakyReluKernelNpu.cpp:28:    cmd.Name("LeakyRelu")
    LeakyReluGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LeakyReluBackwardKernelNpu.cpp:29:    cmd.Name("LeakyReluGrad")
    LeftShift
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/__Lshift__KernelNpu.cpp:28:    cmd.Name("LeftShift").Input(self).Input(other_broadcast).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/__Lshift__KernelNpu.cpp:36:    cmd.Name("LeftShift").Input(self).Input(other_broadcast).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/__iLshift__KernelNpu.cpp:29:    cmd.Name("LeftShift").Input(self).Input(other_broadcast).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/__iLshift__KernelNpu.cpp:37:    cmd.Name("LeftShift").Input(self).Input(other_broadcast).Output(result).Run();
    Lerp
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LerpKernelNpu.cpp:42:    cmd.Name("Lerp")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LerpKernelNpu.cpp:58:    cmd.Name("Lerp")
    LinSpace
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LinspaceKernelNpu.cpp:37:            cmd.Name("LinSpace")
    Log
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/Log10KernelNpu.cpp:28:    cmd.Name("Log")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/Log2KernelNpu.cpp:28:    cmd.Name("Log")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LogKernelNpu.cpp:28:    cmd.Name("Log")
    Log1p
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/Log1pKernelNpu.cpp:28:    cmd.Name("Log1p")
    LogAddExp
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LogAddExp2KernelNpu.cpp:27:  cmd.Name("LogAddExp")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LogAddExpKernelNpu.cpp:27:  cmd.Name("LogAddExp")
    LogMatrixDeterminant
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SlogdetKernelNpu.cpp:29:  cmd.Name("LogMatrixDeterminant")
    LogSigmoid
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LogSigmoidKernelNpu.cpp:30:  cmd.Name("LogSigmoid")
    LogSigmoidGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LogSigmoidBackwardKernelNpu.cpp:32:    cmd.Name("LogSigmoidGrad")
    LogSoftmaxGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LogSoftmaxBackwardKernelNpu.cpp:33:    cmd.Name("LogSoftmaxGrad")
    LogSoftmaxV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/utils/custom_functions/aclops/LogSoftmaxKernelNpu.cpp:26:    cmd.Name("LogSoftmaxV2")
    LogSpaceD
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LogSpaceKernelNpu.cpp:53:    cmd.Name("LogSpaceD")
    LogicalNot
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BitwiseNotKernelNpu.cpp:26:    string real_op_name = (self.dtype() == at::kBool) ? "LogicalNot" : "Invert";
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LogicalNotKernelNpu.cpp:27:  cmd.Name("LogicalNot")
    LogicalOr
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BitwiseOrKernelNpu.cpp:26:    string real_op_name = (self.dtype() == at::kBool) ? "LogicalOr" : "BitwiseOr";
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BitwiseOrKernelNpu.cpp:40:        string real_op_name = (self.dtype() == at::kBool) ? "LogicalOr" : "BitwiseOr";
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LogicalOrKernelNpu.cpp:28:    cmd.Name("LogicalOr").Input(self).Input(other, self.scalar_type()).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LogicalOrKernelNpu.cpp:40:        cmd.Name("LogicalOr").Input(self).Input(other).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/__Ior__KernelNpu.cpp:26:    string real_op_name = (self.dtype() == at::ScalarType::Bool) ? "LogicalOr" : "BitwiseOr";
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/__Ior__KernelNpu.cpp:34:    string real_op_name = (self.dtype() == at::kBool) ? "LogicalOr" : "BitwiseOr";
    LpLoss
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/L1LossKernelNpu.cpp:33:    cmd.Name("LpLoss")
    MaskedFill
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MaskedFillKernelNpu.cpp:43:    cmd.Name("MaskedFill")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MaskedFillKernelNpu.cpp:70:    cmd.Name("MaskedFill")
    MaskedFillRange
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MaskedFillRangeKernelNpu.cpp:58:    cmd.Name("MaskedFillRange")
    MaskedScatter
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MaskedScatterKernelNpu.cpp:36:    cmd.Name("MaskedScatter")
    MatMul
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AddmmKernelNpu.cpp:109:        cmd.Name("MatMul")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MmKernelNpu.cpp:234:        cmd.Name("MatMul")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MvKernelNpu.cpp:34:    cmd.Name("MatMul")
    MatMulV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LinearKernelNpu.cpp:38:    cmd.Name("MatMulV2").Input(input).Input(weight);
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LinearKernelNpu.cpp:52:    cmd.Name("MatMulV2")
    MatrixInverse
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/InverseKernelNpu.cpp:38:    cmd.Name("MatrixInverse")
    MatrixTriangularSolve
~/tmp/pytorch/third_party/op-plugin/op_plugin/utils/custom_functions/aclops/TriangularSolveKernelNpu.cpp:42:    cmd.Name("MatrixTriangularSolve")
    MaxPool3D
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MaxPool3dWithIndicesKernelNpu.cpp:63:    cmd.Name("MaxPool3D")
    MaxPool3DGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MaxPool3dWithIndicesBackwardKernelNpu.cpp:63:    cmd.Name("MaxPool3DGrad")
    MaxPoolGradWithArgmaxV1
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AdaptiveMaxPool2dBackwardKernelNpu.cpp:62:    cmd.Name("MaxPoolGradWithArgmaxV1")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MaxPool2dWithIndicesBackwardKernelNpu.cpp:83:        cmd.Name("MaxPoolGradWithArgmaxV1")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MaxPool2dWithIndicesBackwardKernelNpu.cpp:95:        cmd.Name("MaxPoolGradWithArgmaxV1")
    MaxPoolWithArgmaxV1
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AdaptiveMaxPool2dKernelNpu.cpp:114:    cmd.Name("MaxPoolWithArgmaxV1")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MaxPool2dWithIndicesKernelNpu.cpp:49:    cmd.Name("MaxPoolWithArgmaxV1")
    Maximum
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MaxKernelNpu.cpp:44:    cmd.Name("Maximum").Input(self).Input(other).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MaxKernelNpu.cpp:58:    cmd.Name("Maximum").Input(self).Input(other, self.scalar_type()).Output(result).Run();
    Minimum
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MinKernelNpu.cpp:51:    cmd.Name("Minimum")
    MirrorPad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ReflectionPad2dKernelNpu.cpp:60:    cmd.Name("MirrorPad")
    Mish
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MishKernelNpu.cpp:39:    cmd.Name("Mish")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MishV2KernelNpu.cpp:27:  cmd.Name("Mish")
    MishGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MishBackwardV2KernelNpu.cpp:27:    cmd.Name("MishGrad")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MishKernelNpu.cpp:27:    cmd.Name("MishGrad")
    MlaProlog
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MlaPrologKernelNpu.cpp:37:    cmd.Name("MlaProlog")
    Mod
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FmodKernelNpu.cpp:28:    cmd.Name("Mod")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FmodKernelNpu.cpp:41:    cmd.Name("Mod")
    MseLoss
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MseLossKernelNpu.cpp:36:    cmd.Name("MseLoss")
    MseLossGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MseLossBackwardKernelNpu.cpp:32:  cmd.Name("MseLossGrad")
    MultiHeadAttention
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MultiHeadAttentionKernelNpu.cpp:81:    cmd.Name("MultiHeadAttention")
    MultiHeadAttentionGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MultiHeadAttentionKernelNpu.cpp:173:    cmd.Name("MultiHeadAttentionGrad")
    MultilabelMarginLoss
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MultilabelMarginLossKernelNpu.cpp:33:  cmd.Name("MultilabelMarginLoss")
    MultinomialWithReplacement
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MultinomialKernelNpu.cpp:40:    cmd.Name("MultinomialWithReplacement")
    NLLLoss
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/NLLLoss2dKernelNpu.cpp:62:    cmd.Name("NLLLoss")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/NLLLossKernelNpu.cpp:77:    cmd.Name("NLLLoss")
    NLLLossGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/NLLLoss2dBackwardKernelNpu.cpp:34:    cmd.Name("NLLLossGrad")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/NLLLossBackwardKernelNpu.cpp:45:    cmd.Name("NLLLossGrad")
    NMSWithMask
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/NmsWithMaskKernelNpu.cpp:33:    cmd.Name("NMSWithMask")
    NPUAllocFloatStatus
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FloatStatusKernelNpu.cpp:39:  cmd.Name("NPUAllocFloatStatus")
    NPUClearFloatDebugStatus
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FloatStatusKernelNpu.cpp:95:        cmd.Name("NPUClearFloatDebugStatus")
    NPUClearFloatStatus
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FloatStatusKernelNpu.cpp:89:            cmd.Name("NPUClearFloatStatus")
    NPUClearFloatStatusV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FloatStatusKernelNpu.cpp:86:            cmd.Name("NPUClearFloatStatusV2")
    NPUGetFloatDebugStatus
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FloatStatusKernelNpu.cpp:70:        cmd.Name("NPUGetFloatDebugStatus")
    NPUGetFloatStatus
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FloatStatusKernelNpu.cpp:61:            cmd.Name("NPUGetFloatStatus")
    NPUGetFloatStatusV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FloatStatusKernelNpu.cpp:55:            cmd.Name("NPUGetFloatStatusV2")
    NanToNum
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/NanToNumKernelNpu.cpp:78:    cmd.Name("NanToNum")
    Neg
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/NegKernelNpu.cpp:28:    cmd.Name("Neg")
    NonMaxSuppressionV4
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/NmsV4KernelNpu.cpp:38:    cmd.Name("NonMaxSuppressionV4")
    NonZero
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/WhereKernelNpu.cpp:51:  cmd.Name("NonZero")
    NormalizeBatch
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/NormalizeBatchKernelNpu.cpp:50:  cmd.Name("NormalizeBatch")
    NormalizeV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ImageNormalizeKernelNpu.cpp:40:    cmd.Name("NormalizeV2")
    OneHot
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/OneHotKernelNpu.cpp:66:    cmd.Name("OneHot")
    OneHotD
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/OnehotNpu.cpp:46:    cmd.Name("OneHotD")
    OnesLike
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/OnesLikeKernelNpu.cpp:27:  cmd.Name("OnesLike")
    PRelu
~/tmp/pytorch/third_party/op-plugin/op_plugin/utils/custom_functions/aclops/PreluKernelNpu.cpp:27:    cmd.Name("PRelu")
    PReluGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/utils/custom_functions/aclops/PreluBackwardKernelNpu.cpp:28:    cmd.Name("PReluGrad")
    PSROIPoolingGradV2D
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/PsRoiPoolingKernelNpu.cpp:51:    cmd.Name("PSROIPoolingGradV2D")
    PSROIPoolingV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/PsRoiPoolingKernelNpu.cpp:34:    cmd.Name("PSROIPoolingV2")
    Pack
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/StackKernelNpu.cpp:50:    cmd.Name("Pack");
    Pad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/PadKernelNpu.cpp:31:    cmd.Name("Pad").Input(input).Input(paddings_vector).Output(output).Run();
    PadV3
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ConstantPadNdKernelNpu.cpp:168:            cmd.Name("PadV3")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ConstantPadNdKernelNpu.cpp:177:            cmd.Name("PadV3")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ReplicationPad2dKernelNpu.cpp:40:    cmd.Name("PadV3")
    PadV3Grad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ReflectionPad2dBackwardKernelNpu.cpp:54:    cmd.Name("PadV3Grad")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ReplicationPad2dBackwardKernelNpu.cpp:55:    cmd.Name("PadV3Grad")
    Pdist
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/PdistKernelNpu.cpp:30:    cmd.Name("Pdist")
    Pow
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/Exp2KernelNpu.cpp:28:    cmd.Name("Pow")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/PowKernelNpu.cpp:29:    cmd.Name("Pow")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/PowKernelNpu.cpp:48:        cmd.Name("Pow")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/PowKernelNpu.cpp:61:    cmd.Name("Pow")
    Qr
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LinalgQrKernelNpu.cpp:75:    cmd.Name("Qr")
    QuantConv2D
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/QuantConv2DKernelNpu.cpp:48:    cmd.Name("QuantConv2D").Input(input, "x").Input(weight, "filter").Input(scale, "scale");
    Quantize
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/QuantizeKernelNpu.cpp:59:    cmd.Name("Quantize")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/QuantizePerChannelKernelNpu.cpp:143:    cmd.Name("Quantize")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/QuantizePerChannelKernelNpu.cpp:211:    cmd.Name("Quantize")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/QuantizePerChannelKernelNpu.cpp:58:    cmd.Name("Quantize")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/QuantizePerTensorKernelNpu.cpp:109:    cmd.Name("Quantize")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/QuantizePerTensorKernelNpu.cpp:37:    cmd.Name("Quantize")
    ROIAlign
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RoiAlignKernelNpu.cpp:37:    cmd.Name("ROIAlign")
    ROIAlignGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RoiAlignBackwardKernelNpu.cpp:35:    cmd.Name("ROIAlignGrad")
    RandomChoiceWithMask
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RandomChoiceWithMaskKernelNpu.cpp:42:    cmd.Name("RandomChoiceWithMask")
    Range
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ArangeKernelNpu.cpp:40:    cmd.Name("Range")
    RangeD
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RangeKernelNpu.cpp:41:    cmd.Name("RangeD")
    Reciprocal
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ReciprocalKernelNpu.cpp:29:    cmd.Name("Reciprocal")
    ReduceAll
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AllKernelNpu.cpp:31:    cmd.Name("ReduceAll")
    ReduceAny
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AnyKernelNpu.cpp:30:  cmd.Name("ReduceAny")
    ReduceLogSumExp
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LogSumExpKernelNpu.cpp:53:        cmd.Name("ReduceLogSumExp").Input(self.sub(maxes)).Input(dims).Output(result).Attr("keep_dims", keepdim).Run();
    ReduceMean
~/tmp/pytorch/third_party/op-plugin/op_plugin/utils/custom_functions/aclops/MeanKernelNpu.cpp:44:    cmd.Name("ReduceMean")
    ReduceProd
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ProdKernelNpu.cpp:39:    cmd.Name("ReduceProd").Input(self).Input(dim_list).Output(result).Attr("keep_dims", keepdim).Run();
    ReduceStdV2Update
~/tmp/pytorch/third_party/op-plugin/op_plugin/utils/custom_functions/aclops/VarKernelNpu.cpp:63:    cmd.Name("ReduceStdV2Update")
    ReduceSum
~/tmp/pytorch/third_party/op-plugin/op_plugin/utils/custom_functions/aclops/SumKernelNpu.cpp:36:    cmd.Name("ReduceSum")
    Relu
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AddReluKernelNpu.cpp:32:    cmd.Name("Relu")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ReluKernelNpu.cpp:29:    cmd.Name("Relu")
    ReluGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ThresholdBackwardKernelNpu.cpp:40:        cmd.Name("ReluGrad")
    Renorm
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/EmbeddingRenormKernelNpu.cpp:46:    cmd.Name("Renorm")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RenormKernelNpu.cpp:49:  cmd.Name("Renorm")
    RepeatInterleave
~/tmp/pytorch/third_party/op-plugin/op_plugin/utils/custom_functions/aclops/RepeatInterLeaveKernelNpu.cpp:27:    cmd.Name("RepeatInterleave")
~/tmp/pytorch/third_party/op-plugin/op_plugin/utils/custom_functions/aclops/RepeatInterLeaveKernelNpu.cpp:41:    cmd.Name("RepeatInterleave")
    Resize
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/UpsampleNearest1dKernelNpu.cpp:56:        cmd.Name("Resize")
    ResizeBilinearV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/UpsampleBilinear2dKernelNpu.cpp:40:    cmd.Name("ResizeBilinearV2")
    ResizeBilinearV2Grad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/UpsampleBilinear2dBackwardKernelNpu.cpp:36:    cmd.Name("ResizeBilinearV2Grad")
    ResizeD
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/UpsampleBicubic2dKernelNpu.cpp:54:    cmd.Name("ResizeD")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/UpsampleLinear1dKernelNpu.cpp:75:    cmd.Name("ResizeD")
    ResizeGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/UpsampleNearest1dBackwardKernelNpu.cpp:63:        cmd.Name("ResizeGrad")
    ResizeGradD
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/UpSampleBicubic2dBackwardKernelNpu.cpp:60:    cmd.Name("ResizeGradD")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/UpsampleLinear1dBackwardKernelNpu.cpp:64:    cmd.Name("ResizeGradD")
    ResizeNearestNeighborV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/UpsampleNearest1dKernelNpu.cpp:48:        cmd.Name("ResizeNearestNeighborV2")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/UpsampleNearest2dKernelNpu.cpp:46:    cmd.Name("ResizeNearestNeighborV2")
    ResizeNearestNeighborV2Grad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/UpsampleNearest1dBackwardKernelNpu.cpp:51:        cmd.Name("ResizeNearestNeighborV2Grad")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/UpsampleNearest2dBackwardKernelNpu.cpp:34:    cmd.Name("ResizeNearestNeighborV2Grad")
    ReverseV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FlipKernelNpu.cpp:31:    cmd.Name("ReverseV2")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ReverseKernelNpu.cpp:29:    cmd.Name("ReverseV2")
    RightShift
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/__Rshift__KernelNpu.cpp:29:    cmd.Name("RightShift")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/__Rshift__KernelNpu.cpp:43:    cmd.Name("RightShift")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/__iRshift__KernelNpu.cpp:27:    cmd.Name("RightShift").Input(self).Input(other, self.scalar_type()).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/__iRshift__KernelNpu.cpp:34:    cmd.Name("RightShift").Input(self).Input(other).Output(result).Run();
    RmsNorm
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RmsNormKernelNpu.cpp:49:    cmd.Name("RmsNorm")
    RmsNormGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RmsNormBackwardKernelNpu.cpp:29:    cmd.Name("RmsNormGrad")
    Roll
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RollKernelNpu.cpp:31:    cmd.Name("Roll")
    RotaryMul
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RotaryMulKernelNpu.cpp:33:        cmd.Name("RotaryMul").Input(x).Input(r1).Input(r2).Output(y).Run();
    RotaryMulGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RotaryMulKernelNpu.cpp:77:            cmd.Name("RotaryMulGrad")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RotaryMulKernelNpu.cpp:90:            cmd.Name("RotaryMulGrad")
    RotatedBoxDecode
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RotatedBoxDecodeKernelNpu.cpp:32:    cmd.Name("RotatedBoxDecode").Input(self).Input(deltas).Output(result).Attr("weight", weight_list).Run();
    RotatedBoxEncode
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RotatedBoxEncodeKernelNpu.cpp:34:    cmd.Name("RotatedBoxEncode")
    RotatedIou
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RotatedIouKernelNpu.cpp:36:    cmd.Name("RotatedIou")
    RotatedOverlaps
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RotatedOverlapsKernelNpu.cpp:30:    cmd.Name("RotatedOverlaps")
    Round
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RoundDecimalsKernelNpu.cpp:35:  cmd.Name("Round")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RoundKernelNpu.cpp:34:    cmd.Name("Round")
    Rsqrt
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RsqrtKernelNpu.cpp:28:    cmd.Name("Rsqrt")
    ScaledMaskedSoftmax
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ScaledMaskedSoftmaxKernelNpu.cpp:51:    cmd.Name("ScaledMaskedSoftmax")
    ScaledMaskedSoftmaxGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ScaledMaskedSoftmaxKernelNpu.cpp:32:    cmd.Name("ScaledMaskedSoftmaxGrad")
    Scatter
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ScatterUpdateKernelNpu.cpp:32:    cmd.Name("Scatter")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ScatterUpdateKernelNpu.cpp:53:    cmd.Name("Scatter")
    ScatterElements
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MaxUnpool3dKernelNpu.cpp:54:    cmd.Name("ScatterElements")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ScatterAddKernelNpu.cpp:33:  cmd.Name("ScatterElements")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ScatterKernelNpu.cpp:28:    cmd.Name("ScatterElements")
    ScatterNdAdd
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/PutKernelNpu.cpp:41:    accumulate ? cmd.Name("ScatterNdAdd") : cmd.Name("ScatterNdUpdate");
    ScatterNdUpdate
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/PutKernelNpu.cpp:41:    accumulate ? cmd.Name("ScatterNdAdd") : cmd.Name("ScatterNdUpdate");
    ScatterUpdate
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/EmbeddingRenormKernelNpu.cpp:63:    cmd.Name("ScatterUpdate")
    SearchSorted
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SearchsortedKernelNpu.cpp:32:    cmd.Name("SearchSorted")
    Select
~/tmp/pytorch/third_party/op-plugin/op_plugin/utils/custom_functions/aclops/WhereKernelNpu.cpp:40:    cmd.Name("Select").Input(condition).Input(self_cp).Input(other_cp).Output(out).Run();
    Selu
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SeluKernelNpu.cpp:28:    cmd.Name("Selu")
    SeluGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SeluKernelNpu.cpp:49:    cmd.Name("SeluGrad")
    ShuffleChannel
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ChannelShuffleKernelNpu.cpp:26:    cmd.Name("ShuffleChannel")
    Sigmoid
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SigmoidKernelNpu.cpp:29:    cmd.Name("Sigmoid")
    SigmoidCrossEntropyWithLogitsGradV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BinaryCrossEntropyWithLogitsKernelNpu.cpp:105:  cmd.Name("SigmoidCrossEntropyWithLogitsGradV2")
    SigmoidCrossEntropyWithLogitsV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BinaryCrossEntropyWithLogitsKernelNpu.cpp:61:  cmd.Name("SigmoidCrossEntropyWithLogitsV2")
    SigmoidGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SigmoidBackwardKernelNpu.cpp:32:    cmd.Name("SigmoidGrad")
    Sign
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SignKernelNpu.cpp:26:  cmd.Name("Sign")
    SignBitsPack
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SignBitsPackKernelNpu.cpp:32:    cmd.Name("SignBitsPack")
    SignBitsUnpack
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SignBitsUnpackKernelNpu.cpp:43:    cmd.Name("SignBitsUnpack")
    SilentCheck
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SilentCheckKernelNpu.cpp:29:    cmd.Name("SilentCheck")
    SiluGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SiluBackwardKernelNpu.cpp:30:    cmd.Name("SiluGrad")
    Sin
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SinKernelNpu.cpp:28:    cmd.Name("Sin")
    Sinh
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SinhKernelNpu.cpp:27:  cmd.Name("Sinh")
    Slice
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SliceKernelNpu.cpp:31:    cmd.Name("Slice")
    SmoothL1LossGradV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SmoothL1LossBackwardKernelNpu.cpp:34:    cmd.Name("SmoothL1LossGradV2")
    SmoothL1LossV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SmoothL1LossKernelNpu.cpp:35:    cmd.Name("SmoothL1LossV2")
    SoftMarginLoss
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SoftMarginLossKernelNpu.cpp:36:    cmd.Name("SoftMarginLoss")
    SoftMarginLossGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SoftMarginLossBackwardKernelNpu.cpp:33:  cmd.Name("SoftMarginLossGrad")
    SoftShrink
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SoftShrinkKernelNpu.cpp:30:    cmd.Name("SoftShrink")
    SoftShrinkGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SoftShrinkBackwardKernelNpu.cpp:30:  cmd.Name("SoftShrinkGrad")
    SoftmaxCrossEntropyWithLogits
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SoftmaxCrossEntropyWithLogitsKernelNpu.cpp:30:    cmd.Name("SoftmaxCrossEntropyWithLogits")
    SoftmaxGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SoftmaxBackwardKernelNpu.cpp:32:  cmd.Name("SoftmaxGrad")
    SoftmaxV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SoftmaxKernelNpu.cpp:31:    cmd.Name("SoftmaxV2")
    SoftplusV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SoftplusKernelNpu.cpp:32:    cmd.Name("SoftplusV2")
    SoftplusV2Grad
~/tmp/pytorch/third_party/op-plugin/op_plugin/utils/custom_functions/aclops/SoftplusBackwardKernelNpu.cpp:27:    cmd.Name("SoftplusV2Grad")
    Sort
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ArgsortKernelNpu.cpp:33:    cmd.Name("Sort").Input(self).Output(values).Output(indices).Attr("axis", dim).Attr("descending", descending).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SortKernelNpu.cpp:32:    cmd.Name("Sort")
    SortV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SortWithoutIndicesKernelNpu.cpp:31:    cmd.Name("SortV2")
    Sqrt
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SqrtKernelNpu.cpp:27:    cmd.Name("Sqrt")
    Square
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/PowKernelNpu.cpp:43:        cmd.Name("Square")
    StatelessBernoulli
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BernoulliKernelNpu.cpp:40:        cmd.Name("StatelessBernoulli")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BernoulliKernelNpu.cpp:61:        cmd.Name("StatelessBernoulli")
    StatelessDropOutGenMask
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DropoutGenMaskKernelNpu.cpp:42:    cmd.Name("StatelessDropOutGenMask")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DropoutKernelNpu.cpp:174:  cmd.Name("StatelessDropOutGenMask")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DropoutKernelNpu.cpp:90:  cmd.Name("StatelessDropOutGenMask")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/opapi/FlashAttentionKernelNpuOpApi.cpp:97:    cmd.Name("StatelessDropOutGenMask")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/opapi/FlashAttentionV2KernelNpuOpApi.cpp:87:    cmd.Name("StatelessDropOutGenMask")
    StatelessRandomUniformV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RandomKernelNpu.cpp:48:  cmd.Name("StatelessRandomUniformV2")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/UniformKernelNpu.cpp:43:    cmd.Name("StatelessRandomUniformV2")
    StatelessRandperm
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RandpermKernelNpu.cpp:34:  cmd.Name("StatelessRandperm")
    StrideAdd
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/StrideAddKernelNpu.cpp:32:    cmd.Name("StrideAdd")
    StridedSlice
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/IndexingKernelNpu.cpp:31:    cmd.Name("StridedSlice")
    Sub
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RsubKernelNpu.cpp:40:        cmd.Name("Sub").Input(other).Input(other_mul_result).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RsubKernelNpu.cpp:42:        cmd.Name("Sub").Input(other).Input(self).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RsubKernelNpu.cpp:54:    cmd.Name("Sub").Input(other, self.scalar_type()).Input(scalar_value).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SubKernelNpu.cpp:32:    cmd.Name("Sub").Input(self).Input(scalarValue, self.scalar_type()).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SubKernelNpu.cpp:42:    cmd.Name("Sub").Input(self, other_mul_alpha.scalar_type()).Input(other_mul_alpha).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SubKernelNpu.cpp:61:        cmd.Name("Sub").Expect(unified_result).Input(self).Input(other_mul_result).Output(result).Run();
    SubSample
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SubSampleKernelNpu.cpp:29:    cmd.Name("SubSample")
    Svd
~/tmp/pytorch/third_party/op-plugin/op_plugin/utils/custom_functions/aclops/LinalgSvdKernelNpu.cpp:120:        cmd.Name("Svd")
    Swish
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SiluKernelNpu.cpp:32:    cmd.Name("Swish")
    SwishGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/SiluKernelNpu.cpp:74:    cmd.Name("SwishGrad")
    SyncBatchNormBackwardElemt
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BatchNormBackwardElemtKernelNpu.cpp:80:    cmd.Name("SyncBatchNormBackwardElemt")
    SyncBatchNormBackwardReduce
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BatchNormBackwardReduceKernelNpu.cpp:63:    cmd.Name("SyncBatchNormBackwardReduce")
    SyncBatchNormGatherStats
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BatchNormGatherStatsUpdateKernelNpu.cpp:44:    cmd.Name("SyncBatchNormGatherStats")
    Tan
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/TanKernelNpu.cpp:26:  cmd.Name("Tan")
    Tanh
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/TanhKernelNpu.cpp:27:    cmd.Name("Tanh")
    TanhGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/TanhBackwardKernelNpu.cpp:30:    cmd.Name("TanhGrad")
    ThresholdGradV2D
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ThresholdBackwardKernelNpu.cpp:33:        cmd.Name("ThresholdGradV2D")
    ThresholdV2D
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ThresholdKernelNpu.cpp:30:  cmd.Name("ThresholdV2D")
    Tile
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/RepeatKernelNpu.cpp:30:    cmd.Name("Tile")
    TopKV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/TopKKernelNpu.cpp:34:  cmd.Name("TopKV2")
    Trace
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/TraceKernelNpu.cpp:30:    cmd.Name("Trace")
    Transpose
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/TransposeKernelNpu.cpp:33:        cmd.Name("Transpose").Input(self).Input(perm).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/TransposeKernelNpu.cpp:36:        cmd.Name("Transpose").InputWithoutContiguous(self).Input(perm).Output(result).Run();
    Tril
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/TrilKernelNpu.cpp:27:    cmd.Name("Tril")
    Triu
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/TriuKernelNpu.cpp:27:    cmd.Name("Triu")
    Trunc
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/TruncKernelNpu.cpp:26:    cmd.Name("Trunc").Input(self).Output(result).Run();
    UpsampleNearest3d
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/UpSampleNearest3dKernelNpu.cpp:51:    cmd.Name("UpsampleNearest3d").Input(input).Output(result).Attr("output_size", output_size).Run();
    UpsampleNearest3dGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/UpsampleNearest3dBackwardKernelNpu.cpp:65:    cmd.Name("UpsampleNearest3dGrad")
    UpsampleTrilinear3d
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/UpsampleTrilinear3dKernelNpu.cpp:50:    cmd.Name("UpsampleTrilinear3d")
    UpsampleTrilinear3dGrad
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/UpsampleTrilinear3dBackwardKernelNpu.cpp:64:    cmd.Name("UpsampleTrilinear3dGrad")
    ViewCopy
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ViewcopyKernelNpu.cpp:107:        cmd.Name("ViewCopy")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ViewcopyKernelNpu.cpp:95:        cmd.Name("ViewCopy")
    Xlogy
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/XlogyKernelNpu.cpp:27:    cmd.Name("Xlogy").Input(self).Input(other).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/XlogyKernelNpu.cpp:34:    cmd.Name("Xlogy").Input(self).Input(other, self.scalar_type()).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/XlogyKernelNpu.cpp:41:    cmd.Name("Xlogy").Input(self, other.scalar_type()).Input(other).Output(result).Run();
    YoloBoxesEncode
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/YoloBoxesEncodeKernelNpu.cpp:70:    cmd.Name("YoloBoxesEncode")



READY:

A
    Abs
~/tmp/pytorch/third_party/op-plugin/op_plugin/config/README.md:97:    cmd.Name("Abs")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AbsKernelNpu.cpp:27:    cmd.Name("Abs")
    Add
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AddKernelNpu.cpp:59:    cmd.Name("Add")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/AddKernelNpu.cpp:94:            cmd.Name("Add").Input(self).Input(other).Output(result, "", c10::nullopt, real_type).Run();
B
C
    Ceil
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/CeilKernelNpu.cpp:26:    cmd.Name("Ceil").Input(self).Output(result).Run();
D
E
    Equal
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/EqKernelNpu.cpp:31:    cmd.Name("Equal")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/EqKernelNpu.cpp:46:    cmd.Name("Equal")
F
    Fill
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FillKernelNpu.cpp:27:    cmd.Name("Fill");
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/FillKernelNpu.cpp:43:    cmd.Name("Fill");
G
    Greater
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/GtKernelNpu.cpp:29:  cmd.Name("Greater")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/GtKernelNpu.cpp:41:  cmd.Name("Greater")
    GreaterEqual
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/GeKernelNpu.cpp:28:  cmd.Name("GreaterEqual")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/GeKernelNpu.cpp:39:  cmd.Name("GreaterEqual")
H
I
    IsFinite
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/IsfiniteKernelNpu.cpp:37:    cmd.Name("IsFinite")
J
K
L
    Less
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LtKernelNpu.cpp:29:  cmd.Name("Less")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LtKernelNpu.cpp:40:  cmd.Name("Less")
    LessEqual
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LeKernelNpu.cpp:27:  cmd.Name("LessEqual")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LeKernelNpu.cpp:39:  cmd.Name("LessEqual")
    LogicalAnd
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BitwiseAndKernelNpu.cpp:26:    string real_op_name = (self.dtype() == at::kBool) ? "LogicalAnd" : "BitwiseAnd";
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/BitwiseAndKernelNpu.cpp:40:        string real_op_name = (self.dtype() == at::kBool) ? "LogicalAnd" : "BitwiseAnd";
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LogicalAndKernelNpu.cpp:29:    cmd.Name("LogicalAnd").Input(self_copy).Input(other, self_copy.scalar_type()).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LogicalAndKernelNpu.cpp:46:        cmd.Name("LogicalAnd").Input(self_copy).Input(other_copy).Output(result).Run();
M
    MaskedSelect
./third_party/op-plugin/op_plugin/ops/aclops/MaskedSelectKernelNpu.cpp:50:        .Name("MaskedSelect")
    Mul
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MulKernelNpu.cpp:29:  cmd.Name("Mul")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MulKernelNpu.cpp:44:    cmd.Name("Mul")
N
    NotEqual
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LogicalXorkernelNpu.cpp:29:    cmd.Name("NotEqual").Input(self_copy).Input(other, self_copy.scalar_type()).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/LogicalXorkernelNpu.cpp:46:        cmd.Name("NotEqual").Input(selfCopy).Input(otherCopy).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/NeKernelNpu.cpp:34:    cmd.Name("NotEqual")
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/NeKernelNpu.cpp:51:    cmd.Name("NotEqual")
O
P
Q
R
    RealDiv
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DivKernelNpu.cpp:28:    cmd.Name("RealDiv").Input(self).Input(other, self.scalar_type()).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DivKernelNpu.cpp:36:    cmd.Name("RealDiv").Input(self, other.scalar_type()).Input(other).Output(result).Run();
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/DivKernelNpu.cpp:49:        cmd.Name("RealDiv").Input(self).Input(other).Output(result).Run();
    ReduceMax
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MaxKernelNpu.cpp:51:    cmd.Name("ReduceMax").Input(self).Input(dims).Output(result).Attr("keep_dims", keepdim).Run();
    ReduceMin
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/MinKernelNpu.cpp:66:    cmd.Name("ReduceMin")
S
    StatelessRandomNormalV2
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/NormalKernelNpu.cpp:41:    cmd.Name("StatelessRandomNormalV2")
T
    TensorEqual
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/EqualKernelNpu.cpp:43:    cmd.Name("TensorEqual")
U
V
W
X
Y
Z
    ZerosLike
~/tmp/pytorch/third_party/op-plugin/op_plugin/ops/aclops/ZerosLikeKernelNpu.cpp:27:    cmd.Name("ZerosLike")
COMMENT
