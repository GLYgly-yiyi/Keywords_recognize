// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2018.2
// Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module elu3 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        output_r_address0,
        output_r_ce0,
        output_r_we0,
        output_r_d0,
        cnn_2_out_address0,
        cnn_2_out_ce0,
        cnn_2_out_q0
);

parameter    ap_ST_fsm_state1 = 28'd1;
parameter    ap_ST_fsm_state2 = 28'd2;
parameter    ap_ST_fsm_state3 = 28'd4;
parameter    ap_ST_fsm_state4 = 28'd8;
parameter    ap_ST_fsm_state5 = 28'd16;
parameter    ap_ST_fsm_state6 = 28'd32;
parameter    ap_ST_fsm_state7 = 28'd64;
parameter    ap_ST_fsm_state8 = 28'd128;
parameter    ap_ST_fsm_state9 = 28'd256;
parameter    ap_ST_fsm_state10 = 28'd512;
parameter    ap_ST_fsm_state11 = 28'd1024;
parameter    ap_ST_fsm_state12 = 28'd2048;
parameter    ap_ST_fsm_state13 = 28'd4096;
parameter    ap_ST_fsm_state14 = 28'd8192;
parameter    ap_ST_fsm_state15 = 28'd16384;
parameter    ap_ST_fsm_state16 = 28'd32768;
parameter    ap_ST_fsm_state17 = 28'd65536;
parameter    ap_ST_fsm_state18 = 28'd131072;
parameter    ap_ST_fsm_state19 = 28'd262144;
parameter    ap_ST_fsm_state20 = 28'd524288;
parameter    ap_ST_fsm_state21 = 28'd1048576;
parameter    ap_ST_fsm_state22 = 28'd2097152;
parameter    ap_ST_fsm_state23 = 28'd4194304;
parameter    ap_ST_fsm_state24 = 28'd8388608;
parameter    ap_ST_fsm_state25 = 28'd16777216;
parameter    ap_ST_fsm_state26 = 28'd33554432;
parameter    ap_ST_fsm_state27 = 28'd67108864;
parameter    ap_ST_fsm_state28 = 28'd134217728;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
output  [11:0] output_r_address0;
output   output_r_ce0;
output   output_r_we0;
output  [31:0] output_r_d0;
output  [11:0] cnn_2_out_address0;
output   cnn_2_out_ce0;
input  [31:0] cnn_2_out_q0;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg output_r_ce0;
reg output_r_we0;
reg cnn_2_out_ce0;

(* fsm_encoding = "none" *) reg   [27:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
wire   [4:0] k_5_fu_147_p2;
reg   [4:0] k_5_reg_279;
wire    ap_CS_fsm_state2;
wire   [8:0] tmp_64_fu_173_p2;
reg   [8:0] tmp_64_reg_284;
wire   [0:0] exitcond2_fu_141_p2;
wire   [3:0] i_6_fu_185_p2;
reg   [3:0] i_6_reg_292;
wire    ap_CS_fsm_state3;
wire   [12:0] tmp_105_cast_fu_200_p3;
reg   [12:0] tmp_105_cast_reg_297;
wire   [0:0] exitcond1_fu_179_p2;
wire   [4:0] j_5_fu_214_p2;
reg   [4:0] j_5_reg_305;
wire    ap_CS_fsm_state4;
wire   [0:0] exitcond_fu_208_p2;
reg   [11:0] output_addr_reg_315;
reg   [31:0] cnn_2_out_load_reg_320;
wire    ap_CS_fsm_state5;
wire   [63:0] tmp_32_fu_123_p1;
reg   [63:0] tmp_32_reg_331;
wire    ap_CS_fsm_state6;
wire   [0:0] tmp_16_fu_270_p2;
wire   [63:0] grp_fu_136_p2;
reg   [63:0] tmp_33_reg_336;
wire    ap_CS_fsm_state21;
wire   [63:0] grp_fu_131_p2;
reg   [63:0] tmp_34_reg_341;
wire    ap_CS_fsm_state26;
wire   [31:0] tmp_35_fu_120_p1;
wire    ap_CS_fsm_state27;
reg   [4:0] k_reg_77;
reg   [3:0] i_reg_88;
reg   [4:0] j_reg_99;
wire    ap_CS_fsm_state28;
reg   [31:0] storemerge_reg_110;
wire   [63:0] tmp_106_cast_fu_229_p1;
wire    ap_CS_fsm_state22;
wire    ap_CS_fsm_state7;
wire   [6:0] tmp_s_fu_161_p3;
wire   [8:0] tmp_1_fu_153_p3;
wire   [8:0] p_shl1_cast_fu_169_p1;
wire   [8:0] tmp_cast_fu_191_p1;
wire   [8:0] tmp_65_fu_195_p2;
wire   [12:0] tmp_31_cast_fu_220_p1;
wire   [12:0] tmp_66_fu_224_p2;
wire   [31:0] cnn_2_out_load_to_in_fu_235_p1;
wire   [7:0] tmp_12_fu_238_p4;
wire   [22:0] tmp_1_6_fu_248_p1;
wire   [0:0] notrhs_fu_258_p2;
wire   [0:0] notlhs_fu_252_p2;
wire   [0:0] tmp_14_fu_264_p2;
wire   [0:0] tmp_15_fu_126_p2;
reg   [27:0] ap_NS_fsm;

// power-on initialization
initial begin
#0 ap_CS_fsm = 28'd1;
end

keywords_fptrunc_eOg #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 64 ),
    .dout_WIDTH( 32 ))
keywords_fptrunc_eOg_U90(
    .din0(tmp_34_reg_341),
    .dout(tmp_35_fu_120_p1)
);

keywords_fpext_32fYi #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 32 ),
    .dout_WIDTH( 64 ))
keywords_fpext_32fYi_U91(
    .din0(cnn_2_out_load_reg_320),
    .dout(tmp_32_fu_123_p1)
);

keywords_fcmp_32ng8j #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 32 ),
    .dout_WIDTH( 1 ))
keywords_fcmp_32ng8j_U92(
    .din0(cnn_2_out_load_reg_320),
    .din1(32'd0),
    .opcode(5'd3),
    .dout(tmp_15_fu_126_p2)
);

keywords_dadd_64nhbi #(
    .ID( 1 ),
    .NUM_STAGE( 5 ),
    .din0_WIDTH( 64 ),
    .din1_WIDTH( 64 ),
    .dout_WIDTH( 64 ))
keywords_dadd_64nhbi_U93(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_33_reg_336),
    .din1(64'd13830554455654793216),
    .ce(1'b1),
    .dout(grp_fu_131_p2)
);

keywords_dexp_64nibs #(
    .ID( 1 ),
    .NUM_STAGE( 15 ),
    .din0_WIDTH( 64 ),
    .din1_WIDTH( 64 ),
    .dout_WIDTH( 64 ))
keywords_dexp_64nibs_U94(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(64'd0),
    .din1(tmp_32_reg_331),
    .ce(1'b1),
    .dout(grp_fu_136_p2)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (((exitcond_fu_208_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state4))) begin
        i_reg_88 <= i_6_reg_292;
    end else if (((exitcond2_fu_141_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        i_reg_88 <= 4'd0;
    end
end

always @ (posedge ap_clk) begin
    if (((exitcond1_fu_179_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state3))) begin
        j_reg_99 <= 5'd0;
    end else if ((1'b1 == ap_CS_fsm_state28)) begin
        j_reg_99 <= j_5_reg_305;
    end
end

always @ (posedge ap_clk) begin
    if (((exitcond1_fu_179_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state3))) begin
        k_reg_77 <= k_5_reg_279;
    end else if (((ap_start == 1'b1) & (1'b1 == ap_CS_fsm_state1))) begin
        k_reg_77 <= 5'd0;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state6) & (tmp_16_fu_270_p2 == 1'd1))) begin
        storemerge_reg_110 <= cnn_2_out_load_reg_320;
    end else if ((1'b1 == ap_CS_fsm_state27)) begin
        storemerge_reg_110 <= tmp_35_fu_120_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state5)) begin
        cnn_2_out_load_reg_320 <= cnn_2_out_q0;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        i_6_reg_292 <= i_6_fu_185_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        j_5_reg_305 <= j_5_fu_214_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        k_5_reg_279 <= k_5_fu_147_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((exitcond_fu_208_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4))) begin
        output_addr_reg_315 <= tmp_106_cast_fu_229_p1;
    end
end

always @ (posedge ap_clk) begin
    if (((exitcond1_fu_179_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state3))) begin
        tmp_105_cast_reg_297[12 : 4] <= tmp_105_cast_fu_200_p3[12 : 4];
    end
end

always @ (posedge ap_clk) begin
    if (((tmp_16_fu_270_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state6))) begin
        tmp_32_reg_331 <= tmp_32_fu_123_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state21)) begin
        tmp_33_reg_336 <= grp_fu_136_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state26)) begin
        tmp_34_reg_341 <= grp_fu_131_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((exitcond2_fu_141_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        tmp_64_reg_284[8 : 2] <= tmp_64_fu_173_p2[8 : 2];
    end
end

always @ (*) begin
    if ((((exitcond2_fu_141_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2)) | ((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1)))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = 1'b0;
    end
end

always @ (*) begin
    if (((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((exitcond2_fu_141_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        cnn_2_out_ce0 = 1'b1;
    end else begin
        cnn_2_out_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state28)) begin
        output_r_ce0 = 1'b1;
    end else begin
        output_r_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state28)) begin
        output_r_we0 = 1'b1;
    end else begin
        output_r_we0 = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((ap_start == 1'b1) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            if (((exitcond2_fu_141_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
        end
        ap_ST_fsm_state3 : begin
            if (((exitcond1_fu_179_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state3))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end
        end
        ap_ST_fsm_state4 : begin
            if (((exitcond_fu_208_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state4))) begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end
        end
        ap_ST_fsm_state5 : begin
            ap_NS_fsm = ap_ST_fsm_state6;
        end
        ap_ST_fsm_state6 : begin
            if (((1'b1 == ap_CS_fsm_state6) & (tmp_16_fu_270_p2 == 1'd1))) begin
                ap_NS_fsm = ap_ST_fsm_state28;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state7;
            end
        end
        ap_ST_fsm_state7 : begin
            ap_NS_fsm = ap_ST_fsm_state8;
        end
        ap_ST_fsm_state8 : begin
            ap_NS_fsm = ap_ST_fsm_state9;
        end
        ap_ST_fsm_state9 : begin
            ap_NS_fsm = ap_ST_fsm_state10;
        end
        ap_ST_fsm_state10 : begin
            ap_NS_fsm = ap_ST_fsm_state11;
        end
        ap_ST_fsm_state11 : begin
            ap_NS_fsm = ap_ST_fsm_state12;
        end
        ap_ST_fsm_state12 : begin
            ap_NS_fsm = ap_ST_fsm_state13;
        end
        ap_ST_fsm_state13 : begin
            ap_NS_fsm = ap_ST_fsm_state14;
        end
        ap_ST_fsm_state14 : begin
            ap_NS_fsm = ap_ST_fsm_state15;
        end
        ap_ST_fsm_state15 : begin
            ap_NS_fsm = ap_ST_fsm_state16;
        end
        ap_ST_fsm_state16 : begin
            ap_NS_fsm = ap_ST_fsm_state17;
        end
        ap_ST_fsm_state17 : begin
            ap_NS_fsm = ap_ST_fsm_state18;
        end
        ap_ST_fsm_state18 : begin
            ap_NS_fsm = ap_ST_fsm_state19;
        end
        ap_ST_fsm_state19 : begin
            ap_NS_fsm = ap_ST_fsm_state20;
        end
        ap_ST_fsm_state20 : begin
            ap_NS_fsm = ap_ST_fsm_state21;
        end
        ap_ST_fsm_state21 : begin
            ap_NS_fsm = ap_ST_fsm_state22;
        end
        ap_ST_fsm_state22 : begin
            ap_NS_fsm = ap_ST_fsm_state23;
        end
        ap_ST_fsm_state23 : begin
            ap_NS_fsm = ap_ST_fsm_state24;
        end
        ap_ST_fsm_state24 : begin
            ap_NS_fsm = ap_ST_fsm_state25;
        end
        ap_ST_fsm_state25 : begin
            ap_NS_fsm = ap_ST_fsm_state26;
        end
        ap_ST_fsm_state26 : begin
            ap_NS_fsm = ap_ST_fsm_state27;
        end
        ap_ST_fsm_state27 : begin
            ap_NS_fsm = ap_ST_fsm_state28;
        end
        ap_ST_fsm_state28 : begin
            ap_NS_fsm = ap_ST_fsm_state4;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state21 = ap_CS_fsm[32'd20];

assign ap_CS_fsm_state22 = ap_CS_fsm[32'd21];

assign ap_CS_fsm_state26 = ap_CS_fsm[32'd25];

assign ap_CS_fsm_state27 = ap_CS_fsm[32'd26];

assign ap_CS_fsm_state28 = ap_CS_fsm[32'd27];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

assign ap_CS_fsm_state6 = ap_CS_fsm[32'd5];

assign ap_CS_fsm_state7 = ap_CS_fsm[32'd6];

assign cnn_2_out_address0 = tmp_106_cast_fu_229_p1;

assign cnn_2_out_load_to_in_fu_235_p1 = cnn_2_out_load_reg_320;

assign exitcond1_fu_179_p2 = ((i_reg_88 == 4'd12) ? 1'b1 : 1'b0);

assign exitcond2_fu_141_p2 = ((k_reg_77 == 5'd21) ? 1'b1 : 1'b0);

assign exitcond_fu_208_p2 = ((j_reg_99 == 5'd16) ? 1'b1 : 1'b0);

assign i_6_fu_185_p2 = (i_reg_88 + 4'd1);

assign j_5_fu_214_p2 = (j_reg_99 + 5'd1);

assign k_5_fu_147_p2 = (k_reg_77 + 5'd1);

assign notlhs_fu_252_p2 = ((tmp_12_fu_238_p4 != 8'd255) ? 1'b1 : 1'b0);

assign notrhs_fu_258_p2 = ((tmp_1_6_fu_248_p1 == 23'd0) ? 1'b1 : 1'b0);

assign output_r_address0 = output_addr_reg_315;

assign output_r_d0 = storemerge_reg_110;

assign p_shl1_cast_fu_169_p1 = tmp_s_fu_161_p3;

assign tmp_105_cast_fu_200_p3 = {{tmp_65_fu_195_p2}, {4'd0}};

assign tmp_106_cast_fu_229_p1 = tmp_66_fu_224_p2;

assign tmp_12_fu_238_p4 = {{cnn_2_out_load_to_in_fu_235_p1[30:23]}};

assign tmp_14_fu_264_p2 = (notrhs_fu_258_p2 | notlhs_fu_252_p2);

assign tmp_16_fu_270_p2 = (tmp_15_fu_126_p2 & tmp_14_fu_264_p2);

assign tmp_1_6_fu_248_p1 = cnn_2_out_load_to_in_fu_235_p1[22:0];

assign tmp_1_fu_153_p3 = {{k_reg_77}, {4'd0}};

assign tmp_31_cast_fu_220_p1 = j_reg_99;

assign tmp_64_fu_173_p2 = (tmp_1_fu_153_p3 - p_shl1_cast_fu_169_p1);

assign tmp_65_fu_195_p2 = (tmp_cast_fu_191_p1 + tmp_64_reg_284);

assign tmp_66_fu_224_p2 = (tmp_105_cast_reg_297 + tmp_31_cast_fu_220_p1);

assign tmp_cast_fu_191_p1 = i_reg_88;

assign tmp_s_fu_161_p3 = {{k_reg_77}, {2'd0}};

always @ (posedge ap_clk) begin
    tmp_64_reg_284[1:0] <= 2'b00;
    tmp_105_cast_reg_297[3:0] <= 4'b0000;
end

endmodule //elu3
