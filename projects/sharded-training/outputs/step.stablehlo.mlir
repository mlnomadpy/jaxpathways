module @jit_local attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["data"=4]>
  func.func public @main(%arg0: tensor<1024x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {}]>}, %arg1: tensor<1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}]>}, %arg2: tensor<1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}]>}, %arg3: tensor<64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg4: tensor<64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> (tensor<64xf32> {jax.result_info = "result[0]"}, tensor<64xf32> {jax.result_info = "result[1]"}, tensor<f32> {jax.result_info = "result[2]"}, tensor<64xf32> {jax.result_info = "result[3]"}) {
    %0:4 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4) in_shardings=[<@mesh, [{"data"}, {}]>, <@mesh, [{"data"}]>, <@mesh, [{"data"}]>, <@mesh, [{}]>, <@mesh, [{}]>] out_shardings=[<@mesh, [{}]>, <@mesh, [{}]>, <@mesh, []>, <@mesh, [{}]>] manual_axes={"data"} (%arg5: tensor<256x64xf32>, %arg6: tensor<256xf32>, %arg7: tensor<256xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>) {
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1 = stablehlo.reduce(%arg7 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg10: tensor<f32>, %arg11: tensor<f32>):
        %24 = stablehlo.add %arg10, %arg11 : tensor<f32>
        stablehlo.return %24 : tensor<f32>
      }) : (tensor<f32>) -> tensor<f32>
      %3 = stablehlo.dot_general %arg5, %arg8, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x64xf32>, tensor<64xf32>) -> tensor<256xf32>
      %4 = stablehlo.subtract %3, %arg6 : tensor<256xf32>
      %5 = stablehlo.transpose %arg5, dims = [1, 0] : (tensor<256x64xf32>) -> tensor<64x256xf32>
      %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %7 = stablehlo.multiply %6, %5 : tensor<64x256xf32>
      %8 = stablehlo.multiply %4, %arg7 : tensor<256xf32>
      %9 = stablehlo.dot_general %7, %8, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x256xf32>, tensor<256xf32>) -> tensor<64xf32>
      %10 = "stablehlo.all_reduce"(%9) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg10: tensor<f32>, %arg11: tensor<f32>):
        %24 = stablehlo.add %arg10, %arg11 : tensor<f32>
        stablehlo.return %24 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %11 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %12 = stablehlo.divide %10, %11 : tensor<64xf32>
      %13 = stablehlo.multiply %4, %4 : tensor<256xf32>
      %14 = stablehlo.multiply %arg7, %13 : tensor<256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %15 = stablehlo.reduce(%14 init: %cst_1) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      %16 = "stablehlo.all_reduce"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg10: tensor<f32>, %arg11: tensor<f32>):
        %24 = stablehlo.add %arg10, %arg11 : tensor<f32>
        stablehlo.return %24 : tensor<f32>
      }) : (tensor<f32>) -> tensor<f32>
      %17 = stablehlo.divide %16, %2 : tensor<f32>
      %cst_2 = stablehlo.constant dense<8.000000e-01> : tensor<f32>
      %18 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %19 = stablehlo.multiply %18, %arg9 : tensor<64xf32>
      %20 = stablehlo.add %19, %12 : tensor<64xf32>
      %cst_3 = stablehlo.constant dense<3.000000e-02> : tensor<f32>
      %21 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %22 = stablehlo.multiply %21, %20 : tensor<64xf32>
      %23 = stablehlo.subtract %arg8, %22 : tensor<64xf32>
      sdy.return %23, %20, %17, %12 : tensor<64xf32>, tensor<64xf32>, tensor<f32>, tensor<64xf32>
    } : (tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>, tensor<f32>, tensor<64xf32>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<64xf32>, tensor<64xf32>, tensor<f32>, tensor<64xf32>
  }
}
