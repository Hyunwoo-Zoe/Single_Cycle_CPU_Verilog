///////////////////////////////////////////////////////////////////////////
// MODULE: CPU for TSC microcomputer: cpu.v
// Author: 
// Description: 

// DEFINITIONS
`define WORD_SIZE 16    // data and address word size

// MODULE DECLARATION
module cpu (
    output readM,                       // read from memory
    output [`WORD_SIZE-1:0] address,    // current address for data
    inout [`WORD_SIZE-1:0] data,        // data being input or output
    input inputReady,                   // indicates that data is ready from the input port
    input reset_n,                      // active-low RESET signal
    input clk,                          // clock signal
  
    // for debuging/testing purpose
    output [`WORD_SIZE-1:0] num_inst,   // number of instruction during execution
    output [`WORD_SIZE-1:0] output_port // this will be used for a "WWD" instruction
);
  
wire [3:0] opcode; // IR의 4-bit opcode field
wire [5:0] func;  // IR의 6-bit function field
wire cu_RegWrite; // control unit -> datapath : 레지스터 쓰기 제어 신호
wire cu_ALUSrc; // control unit -> datapath : ALU B 입력 선택 (MuxB / 레지스터값으로 할지 아니면 immediate 값으로 할지)  
wire cu_LHISel; // control unit -> datapath : LHI 명령의 여부
wire cu_PCSrc;  // control unit -> datapath : PC의 branch 여부를 선택  
wire cu_DestSel; // control unit -> datapath : Destination Register 선택 (rd vs rt)   
wire cu_WWD; // control unit -> datapath : WWD instruction 여부 결정 (output_port로 쓰기)

datapath DP (
    .readM (readM),
    .address (address),
    .data (data),
    .clk (clk),
    .reset_n (reset_n),
    .inputReady (inputReady),
    .RegWrite_i (cu_RegWrite),
    .ALUSrc_i (cu_ALUSrc),
    .LHISel_i (cu_LHISel),
    .PCSrc_i (cu_PCSrc),
    .DestSel_i (cu_DestSel),
    .WWD_i (cu_WWD),
    .opcode_o (opcode),
    .func_o (func),
    .num_inst_o (num_inst),
    .output_port_o (output_port)
    );

control_unit CU (
    .opcode (opcode), // datapath에서 추출된 opcode 입력
    .func (func), // datapath에서 추출된 function 입력

    .RegWrite_o (cu_RegWrite), // CU -> datapath: Register File 쓰기 제어
    .ALUSrc_o (cu_ALUSrc), // CU -> datapath: ALU의 SourceB 선택 
    .LHISel_o (cu_LHISel), // CU -> datapath: LHI 명령 여부 제어
    .PCSrc_o (cu_PCSrc), // CU -> datapath: Branch/Jump Instruction 수행 여부 제어
    .DestSel_o (cu_DestSel), // CU -> datapath: Destination Register 선택 제어
    .WWD_o (cu_WWD) // CU -> datapath: WWD 명령 여부 제어
    );

endmodule

///////////////////////////////////////////// Control Unit Instance CU와 Datapath의 Instance DP를 포함한 "CPU" Module /////////////////////////////////////////////////////

module control_unit (
    input [3:0] opcode, // IR의 opcode field 입력
    input [5:0] func, // IR의 func field 입력
    // IR을 바탕으로 Instruction Decoding 하고 Control Signal Generation 진행

    output reg RegWrite_o, // datapath로 전달될 레지스터 쓰기 신호
    output reg ALUSrc_o, // ALU 두 번쨰 입력 선택 (SourceB vs Immediate)
    output reg LHISel_o, // LHI instruction인지 여부 선택 (1이면 LHI)
    output reg PCSrc_o, // Branch Instruction 여부 (1: PC <= jump target)
    output reg DestSel_o, // Destination Register 선택 (0: rd, 1: rt)
    // IR에서 보통 rs, rt, rd 순서로 나와있음 / 0이면 R-type Inst, 1이면 I-type
    output reg WWD_o // WWD 명령 여부 (1: output_port 에 값 전달)
);

always @ * begin // opcode, func가 변할 때마다 즉시 실행
    RegWrite_o = 0; // 기본값: Register Write 비활성화
    ALUSrc_o = 0; // 기본값: ALU B 입력 = Register 데이터로 초기화
    LHISel_o = 0; // 기본값: LHI 사용 안 하는것으로 초기화
    PCSrc_o = 0; // 기본값: PC = PC + 1 (분기 안 하는 것으로 초기화)
    DestSel_o = 0; // 기본값: Destination Register = rd 로 초기화
    WWD_o = 0; // 기본값: WWD 비활성화

    case (opcode) // opcode에 따라 instruction 그룹 결정
        4'hF: begin  // R-type(0xF로 시작) -> func 필드로 세분화해서 결정
            case (func) 
                6'd0 : begin // 6자리 func field가 0이면 ADD
                    RegWrite_o = 1; // RegWrite 활성화
                    DestSel_o  = 0; // Destination Register는 rd
                    end

                6'd28 : WWD_o = 1; // 6자리 func field가 28이면 WWD
                endcase
            end

        4'h4: begin // opcode가 4 이면 ADI (I-type)             
            RegWrite_o = 1; // RegWrite 활성화
            ALUSrc_o   = 1; // Immediate 선택
            DestSel_o  = 1; // Destination Register는 rt
            end

        4'h6: begin  // opcode가 6이면 LHI II-type)
            RegWrite_o = 1; // RegWrite 활성화
            LHISel_o   = 1; // LHI 활성화
            DestSel_o  = 1; // I-type이기에 Desitnation Register는 rt
            end

        4'h9: PCSrc_o = 1; // opcode가 9이면 Jump (J-type)
        endcase
    end
endmodule

////////////////////////////////////////////////////////////////// Control Unit Module 정의 /////////////////////////////////////////////////////////////////////////////

module datapath (
    output readM, // memory read 요청 신호 (항상 1)
    output [15:0] address, // 메모리 주소 출력 (PC 값을 의미함)
    inout [15:0] data,  // 메모리로부터 데이터를 읽거나 쓰는 양방향 버스

    input clk, // clock 신호
    input reset_n, // 비동기 Reset 입력 (0일 때 초기화)
    input inputReady, // 메모리로부터 데이터가 도착했음을 알리는 신호

    input RegWrite_i, // control → datapath: Register Write 제어 신호
    input ALUSrc_i, // control → datapath: ALU 두 번째 입력 선택
    input LHISel_i, // control → datapath: LHI 명령 여부 제어 신호
    input PCSrc_i, // control → datapath: PC branch 수행 여부 제어 신호
    input DestSel_i, // control → datapath: Destination Register 선택
    input WWD_i, // control → datapath: WWD 명령 여부

    output [3:0] opcode_o, // datapath -> control unit: instruction의 opcode 필드 출력
    output [5:0] func_o, // datapath -> control unit: instruction의 function 필드 출력

    output [15:0] num_inst_o, // 실행된 명령어의 수 출력
    output [15:0] output_port_o // 출력 포트 결과값 (WWD 실행 시)
);

reg [15:0] PC, IR; // PC, IR(현재 명령어 레지스터)
reg IR_valid; // IR이 유효한 Instruction을 담고 있는지 여부를 확인
reg [15:0] fetch_buffer; // 메모리에서 읽어온 명령어 데이터를 임시 저장하는 buffer
reg fetch_toggle; // fetch가 발생했음을 표시하는 Flag
reg load_toggle; // IR로 명령어가 로드되었는지 표시하는 flag
reg [15:0] num_inst_r; // 실행된 명령어 수를 세는 Register
reg [15:0] output_port_r; // WWD 명령어 실행 시 출력할 port 값을 저장

assign num_inst_o = num_inst_r; // 외부로 실행 명령어 수를 전달
assign output_port_o = output_port_r; // 외부로 출력 포트 값 전달
assign readM = 1'b1; // 항상 메모리 읽기 요청 (과제에 Memory Write는 없음)
assign address = PC; // 현재 PC 값을 메모리 주소로 출력
assign data = 16'bz; // data 버스는 입력용, z로 초기화해둠

// 메모리로부터 데이터가 준비됐거나 비동기 리셋 신호가 들어올 때 동작!
always @ (posedge inputReady or negedge reset_n) begin
    if (!reset_n) begin // 리셋이 들어오면
        fetch_toggle <= 1'b0; // fetch 상태 초기화
        fetch_buffer <= 16'd0; // fetch 버퍼 초기화
        end

    else begin // 메모리로부터 데이터가 준비되었으면
        fetch_toggle <= ~fetch_toggle; // 새로운 명령어 도착했다고 flag 표시
        fetch_buffer <= data; // buffer에 새로운 명령어 저장
    end

end

wire [3:0] opcode = IR[15:12]; // IR로 fetch한 명령어를 상위 4 bit opcode로 저장
wire [1:0] rs = IR[11:10]; // IR중에 SourceA Register 필드 추출
wire [1:0] rt = IR[9:8]; // IR중에 SourceB Register 필드 추출
wire [1:0] rd = IR[7:6]; // Destination Register (rd) 필드 추출 (R-type)
wire [7:0] imm8 = IR[7:0]; // Immediate 8-bit 필드 (I-type에서 사용)
wire [11:0] tgt12 = IR[11:0]; // Jump Target 필드 (J-type에서 사용)
wire [5:0] func = IR[5:0]; // R-type 명령의 세부 동작 지정 func 필드

assign opcode_o = opcode; // datapath -> CU: opcode 전달
assign func_o = func; // datapath -> CU: func 전달

wire [1:0]  waddr = DestSel_i ? rt : rd; // Destination Register 결정
wire [15:0] signExtImm = {{8{imm8[7]}}, imm8}; // Immediate 값을 Sign-Extension 해서 16-bit으로 만들어줌
wire [15:0] rf_rdata1, rf_rdata2; // Register FIle에서 읽어온 값
reg [15:0] rf_wdata; // Register File에 Write할 값 (데이터)

always @ * begin
    rf_wdata = LHISel_i ? {imm8, 8'h00} : alu_out; 
    // LHI 명령이면 immediate를 상위 바이트에 넣고 하위는 0
    // 아니면 ALU 결과를 레지스터에 저장 (MUX D의 역할 수행)
end

wire rf_we = RegWrite_i & IR_valid;
// 실제 Register Write 수행 조건
// Control에서 쓰기 명령을 허용했고 Valid한 IR이 들어왔을 때만 Write 실행

RF REGFILE (
.addr1 (rs), // IR의 rs의 값을 addr1에 연결
.addr2 (rt), // IR의 rt의 값을 addr2에 연결 
.addr3 (waddr), // IR의 waddr를 addr3에 연결
.data3 (rf_wdata), // rf_data를 data3에 연결
.write (rf_we), // rf_we를 write에 연결
.clk (clk), // clk 연결
.reset (~reset_n), // reset_n은 active-low이니까 active-high로 만들어줌
.data1 (rf_rdata1), // rf_rdata1에 output data1 전달
.data2 (rf_rdata2) // rf_rdata2에 output data2 전달
);

// alu_B에는 ALUSrc가 1이면 Immediate값을 취하고 아니면 rf_rdata2 (레지스터 값)
wire [15:0] alu_B  = ALUSrc_i ? signExtImm : rf_rdata2;
wire [15:0] alu_out; // alu_out 결과값 전달 wire

ALU ALU_U (
.A (rf_rdata1), // ALU 입력 A에 rf_rdata1 연결
.B (alu_B), // ALU 입력 B에 alu_B 연결
.Cin (1'b0), // Cin에 0으로 초기화 해 둠
.OP (4'b0000), // Opcode는 현재 0000으로(덧셈) 일단 초기화
.Cout(), // Carry Out 출력 (사용하지를 않음)
.C (alu_out) // ALu 결과를 출력 -> rf_wdata에 사용되거나 다음 로직에서 사용
);

wire [15:0] PC_next = PCSrc_i ? {4'b0000, tgt12} : PC + 16'd1;
// PC_next는 PCSrc_i, 즉, branch instruction이면 Jump 대상 주소로 이동, 일반 명령이면 +1 해서 이동

always @ (posedge clk or negedge reset_n) begin
    if (!reset_n) begin // reset이 들어오면 전부 초기화
        PC <= 16'd0; 
        IR_valid <= 1'b0;
        load_toggle <= 1'b0;
        num_inst_r <= 16'd0;
        output_port_r <= 16'd0;
        end

        else begin // posedge clk이 들어오면

        if (fetch_toggle != load_toggle) begin
            IR <= fetch_buffer; // IR에 메모리에서 가져온 명령어 저장
            IR_valid <= 1'b1; // 명령어 유효 flag 설정
            load_toggle<= fetch_toggle; // 토글을 동기화 -> 다음 clk cycle에 중복 로드 방지
        end

        else if (IR_valid) begin // Execute 단계 (이미 fetch 끝난 상태)
            if (WWD_i)
                output_port_r <= rf_rdata1; // Writing to Output 실행

            if (RegWrite_i | WWD_i | PCSrc_i | ALUSrc_i | LHISel_i)
                num_inst_r <= num_inst_r + 16'd1;
                // 실행된 명령어 개수를 카운트 (각 명령어가 끝날 때마다 업데이트)

            PC <= PC_next; // PC값 업데이트
            IR_valid <= 1'b0; // IR_valid값 0으로 초기화
        end
    end
end
endmodule

///////////////////////////////////////////////// RF의 Instance REGFILE과 ALU의 Instance ALU_U를 포함한 "datapath" Module /////////////////////////////////////////////////////

module RF (
    input [1:0]  addr1, addr2, addr3, // rs(sourceA), rt(sourceB), rd(destination)
    input [15:0] data3, // Write할 데이터
    input write, // RF 활성화 신호
    input clk,// Clock
    input reset, // Reset (posedge clk에서 사용)
    output reg [15:0] data1, data2 // Read 출력값 (R[addr1], R[addr2])
);

    reg [15:0] regFile [3:0]; // 4개의 16bit 레지스터 배열을 정의

    always @ (posedge clk) begin
        if (reset) begin  // Reset이 들어오면 모든 레지스터를 초기화        
            regFile[0] <= 16'd0;
            regFile[1] <= 16'd0;
            regFile[2] <= 16'd0;
            regFile[3] <= 16'd0;
        end

        else if (write) // Write가 들어오면
            regFile[addr3] <= data3; // addr3에 data3를 저장
    end

    // 쓰기의 과정

    always @ * begin
        data1 = regFile[addr1]; // Reg[addr1] -> data1
        data2 = regFile[addr2];  // Reg[addr2] -> data2
    end   

    // 읽기의 과정

endmodule


///////////////////////////////////////////////////////////////// Register File "RF" Module ////////////////////////////////////////////////////////////////////////////

module ALU (
    input [15:0] A, B, // ALU의 Source Register Operand     
    input Cin,  // Cin for ALU
    input [3:0] OP, // ALU의 opcode 
    output reg Cout, // Carry Out     
    output reg [15:0] C // 연산 결과 나타내는 C
);
    always @ * begin
        {Cout, C} = 17'd0; // Cout과 C(연산 결과)를 전부 0으로 초기화
        case (OP)
            4'b0000: {Cout, C} = A + B + Cin; // ADD           
            4'b0001: {Cout, C} = {1'b0, A} - ({1'b0, B}+Cin); // borrow를 고려한 SUB
            4'b0010: C = A; // Pass
            4'b0011: C = ~(A & B); // NAND
            4'b0100: C = ~(A | B); // NOR
            4'b0101: C = ~(A ^ B); // XNOR
            4'b0110: C = ~A; // NOT
            4'b0111: C = A & B; // AND
            4'b1000: C = A | B; // OR
            4'b1001: C = A ^ B; // XOR
            4'b1010: C = A >> 1; // Right Shift (Logical)
            4'b1011: C = $signed(A) >>> 1; // Right Shift (Arithmetic)
            4'b1100: C = {A[0], A[15:1]}; // Rotate right by 1
            4'b1101: C = A << 1; // Left Shift (Logical)
            4'b1110: C = A << 1; // Right Shift (Arithmetic)
            4'b1111: C = {A[14:0], A[15]}; // Rotate Left by 1
        endcase
    end
endmodule

///////////////////////////////////////////////////////////////////////// "ALU" Module //////////////////////////////////////////////////////////////////////////////////
