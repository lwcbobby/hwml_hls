
/** Hardware accelerator design of the convolution layer.
 *
 * The design is performing the computation of a single tile.
 *
 * \tparam T scalar type
 * \tparam TH tile output height
 * \tparam TW tile output width
 * \tparam TC tile input channels
 * \tparam TF tile output channels
 * \tparam K kernel size
 */
void conv_layer_tile_accel(int input[TC][TH][TW],
                           int weight[TF][TC][TH][TW],
                           int &output[TF][0][0]) {
#pragma HLS ARRAY_PARTITION VARIABLE = input DIM = 1 COMPLETE
//#pragma HLS ARRAY_PARTITION VARIABLE = weight DIM = 1 COMPLETE
#pragma HLS ARRAY_PARTITION VARIABLE = weight DIM = 2 COMPLETE
#pragma HLS ARRAY_PARTITION VARIABLE = output DIM = 1 COMPLETE

  // initialize output with bias
for (int th = 0; th < TH; th++) {
	for (int tw = 0; tw < TW; tw++) {
#pragma HLS PIPELINE II = 1
		for (int tf = 0; tf < TF; tf++) {
        		output[tf][th][tw] = 0;
      		}
    	}
}

  // run the computation
for (int th = 0; th < TH; th++) {
	for (int tw = 0; tw < TW; tw++) {
#pragma HLS PIPELINE II = 1
		for (int tf = 0; tf < TF; tf++) {
			for (int tc = 0; tc < TC; tc++) {
              			*output[tf][0][0] +=
                  		weight[tf][tc][th][tw] * input[tc][th][tw];
            		}
          	}
        }
}

}
