	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 15, 0	sdk_version 15, 2
	.file	0 "/Users/lmwgsrwny/Documents/_E8/C_Lang_Mamba/src/build" "/Users/lmwgsrwny/Documents/_E8/C_Lang_Mamba/src/main.c" md5 0xd5a662030720032527d968c29cbf3ff8
	.file	1 "/Users/lmwgsrwny/Documents/_E8/C_Lang_Mamba/src" "main.c" md5 0xd5a662030720032527d968c29cbf3ff8
	.globl	_read_input_from_file           ; -- Begin function read_input_from_file
	.p2align	2
_read_input_from_file:                  ; @read_input_from_file
Lfunc_begin0:
	.loc	1 44 0                          ; main.c:44:0
	.cfi_startproc
; %bb.0:
	;DEBUG_VALUE: read_input_from_file:filename <- $x0
	;DEBUG_VALUE: read_input_from_file:input <- $x1
	;DEBUG_VALUE: read_input_from_file:size <- $w2
	sub	sp, sp, #80
	stp	x24, x23, [sp, #16]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #32]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #48]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #64]             ; 16-byte Folded Spill
	add	x29, sp, #64
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	mov	x22, x2
Ltmp0:
	;DEBUG_VALUE: read_input_from_file:size <- $w22
	mov	x21, x1
Ltmp1:
	;DEBUG_VALUE: read_input_from_file:input <- $x21
	mov	x20, x0
Ltmp2:
	;DEBUG_VALUE: read_input_from_file:filename <- $x20
	.loc	1 45 18 prologue_end            ; main.c:45:18
Lloh0:
	adrp	x1, l_.str@PAGE
Lloh1:
	add	x1, x1, l_.str@PAGEOFF
	bl	_fopen
Ltmp3:
	;DEBUG_VALUE: read_input_from_file:file <- $x0
	.loc	1 46 9                          ; main.c:46:9
	cbz	x0, LBB0_7
Ltmp4:
; %bb.1:
	;DEBUG_VALUE: read_input_from_file:file <- $x0
	;DEBUG_VALUE: read_input_from_file:filename <- $x20
	;DEBUG_VALUE: read_input_from_file:input <- $x21
	;DEBUG_VALUE: read_input_from_file:size <- $w22
	.loc	1 0 0 is_stmt 0                 ; main.c:0:0
	mov	x19, x0
Ltmp5:
	;DEBUG_VALUE: read_input_from_file:file <- $x19
	;DEBUG_VALUE: i <- 0
	.loc	1 51 5 is_stmt 1                ; main.c:51:5
	cmp	w22, #1
	b.lt	LBB0_5
Ltmp6:
; %bb.2:
	;DEBUG_VALUE: i <- 0
	;DEBUG_VALUE: read_input_from_file:file <- $x19
	;DEBUG_VALUE: read_input_from_file:filename <- $x20
	;DEBUG_VALUE: read_input_from_file:input <- $x21
	;DEBUG_VALUE: read_input_from_file:size <- $w22
	.loc	1 0 5 is_stmt 0                 ; main.c:0:5
	mov	x23, #0                         ; =0x0
Ltmp7:
	.loc	1 51 23                         ; main.c:51:23
	mov	w24, w22
Ltmp8:
	;DEBUG_VALUE: read_input_from_file:size <- $w24
	.loc	1 0 23                          ; main.c:0:23
Lloh2:
	adrp	x22, l_.str.2@PAGE
Lloh3:
	add	x22, x22, l_.str.2@PAGEOFF
Ltmp9:
LBB0_3:                                 ; =>This Inner Loop Header: Depth=1
	;DEBUG_VALUE: read_input_from_file:size <- $w24
	;DEBUG_VALUE: read_input_from_file:file <- $x19
	;DEBUG_VALUE: read_input_from_file:filename <- $x20
	;DEBUG_VALUE: read_input_from_file:input <- $x21
	;DEBUG_VALUE: i <- $x23
	.loc	1 52 35 is_stmt 1               ; main.c:52:35
	add	x8, x21, x23
	.loc	1 52 13 is_stmt 0               ; main.c:52:13
	str	x8, [sp]
	mov	x0, x19
	mov	x1, x22
	bl	_fscanf
Ltmp10:
	.loc	1 52 13                         ; main.c:52:13
	cmp	w0, #1
	b.ne	LBB0_6
Ltmp11:
; %bb.4:                                ;   in Loop: Header=BB0_3 Depth=1
	;DEBUG_VALUE: i <- $x23
	;DEBUG_VALUE: read_input_from_file:size <- $w24
	;DEBUG_VALUE: read_input_from_file:file <- $x19
	;DEBUG_VALUE: read_input_from_file:filename <- $x20
	;DEBUG_VALUE: read_input_from_file:input <- $x21
	.loc	1 51 32 is_stmt 1               ; main.c:51:32
	add	x23, x23, #1
Ltmp12:
	;DEBUG_VALUE: i <- $x23
	.loc	1 51 5 is_stmt 0                ; main.c:51:5
	cmp	x24, x23
	b.ne	LBB0_3
Ltmp13:
LBB0_5:
	;DEBUG_VALUE: read_input_from_file:file <- $x19
	;DEBUG_VALUE: read_input_from_file:filename <- $x20
	;DEBUG_VALUE: read_input_from_file:input <- $x21
	.loc	1 59 5 is_stmt 1                ; main.c:59:5
	mov	x0, x19
	bl	_fclose
Ltmp14:
	.loc	1 60 5                          ; main.c:60:5
	str	x20, [sp]
Lloh4:
	adrp	x0, l_.str.4@PAGE
Lloh5:
	add	x0, x0, l_.str.4@PAGEOFF
	bl	_printf
Ltmp15:
	.loc	1 61 1 epilogue_begin           ; main.c:61:1
	ldp	x29, x30, [sp, #64]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #48]             ; 16-byte Folded Reload
Ltmp16:
	;DEBUG_VALUE: read_input_from_file:filename <- [DW_OP_LLVM_entry_value 1] $x0
	ldp	x22, x21, [sp, #32]             ; 16-byte Folded Reload
Ltmp17:
	;DEBUG_VALUE: read_input_from_file:input <- [DW_OP_LLVM_entry_value 1] $x1
	ldp	x24, x23, [sp, #16]             ; 16-byte Folded Reload
	add	sp, sp, #80
	ret
Ltmp18:
LBB0_6:
	;DEBUG_VALUE: i <- $x23
	;DEBUG_VALUE: read_input_from_file:size <- $w24
	;DEBUG_VALUE: read_input_from_file:file <- $x19
	;DEBUG_VALUE: read_input_from_file:filename <- $x20
	;DEBUG_VALUE: read_input_from_file:input <- $x21
	.loc	1 53 21                         ; main.c:53:21
Lloh6:
	adrp	x8, ___stderrp@GOTPAGE
Lloh7:
	ldr	x8, [x8, ___stderrp@GOTPAGEOFF]
Lloh8:
	ldr	x0, [x8]
	.loc	1 53 13 is_stmt 0               ; main.c:53:13
	str	x23, [sp]
Lloh9:
	adrp	x1, l_.str.3@PAGE
Lloh10:
	add	x1, x1, l_.str.3@PAGEOFF
	bl	_fprintf
Ltmp19:
	.loc	1 54 13 is_stmt 1               ; main.c:54:13
	mov	x0, x19
	bl	_fclose
Ltmp20:
	.loc	1 55 13                         ; main.c:55:13
	mov	w0, #1                          ; =0x1
	bl	_exit
Ltmp21:
LBB0_7:
	;DEBUG_VALUE: read_input_from_file:file <- $x0
	;DEBUG_VALUE: read_input_from_file:filename <- $x20
	;DEBUG_VALUE: read_input_from_file:input <- $x21
	;DEBUG_VALUE: read_input_from_file:size <- $w22
	.loc	1 47 9                          ; main.c:47:9
	bl	_read_input_from_file.cold.1
Ltmp22:
	.loh AdrpAdd	Lloh0, Lloh1
	.loh AdrpAdd	Lloh2, Lloh3
	.loh AdrpAdd	Lloh4, Lloh5
	.loh AdrpAdd	Lloh9, Lloh10
	.loh AdrpLdrGotLdr	Lloh6, Lloh7, Lloh8
Lfunc_end0:
	.cfi_endproc
	.file	2 "/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk/usr/include" "_stdio.h" md5 0x3d165ce37807d123c2b5ed4c31060b16
	.file	3 "/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk/usr/include/arm" "_types.h" md5 0xb270144f57ae258d0ce80b8f87be068c
	.file	4 "/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk/usr/include/sys" "_types.h" md5 0xaf82ff6119a9fa80fad635d276556d46
	.file	5 "/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk/usr/include" "_printf.h" md5 0x1dbdc20fec1369d4ad4aa87b19eddb1b
	.file	6 "/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk/usr/include" "_stdlib.h" md5 0x1fbf79127e45bcc71ac6e64ce165d68a
                                        ; -- End function
	.globl	_write_output_to_file           ; -- Begin function write_output_to_file
	.p2align	2
_write_output_to_file:                  ; @write_output_to_file
Lfunc_begin1:
	.loc	1 64 0                          ; main.c:64:0
	.cfi_startproc
; %bb.0:
	;DEBUG_VALUE: write_output_to_file:filename <- $x0
	;DEBUG_VALUE: write_output_to_file:output <- $x1
	;DEBUG_VALUE: write_output_to_file:size <- $w2
	sub	sp, sp, #64
	stp	x22, x21, [sp, #16]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #32]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #48]             ; 16-byte Folded Spill
	add	x29, sp, #48
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	mov	x21, x2
Ltmp23:
	;DEBUG_VALUE: write_output_to_file:size <- $w21
	mov	x19, x1
Ltmp24:
	;DEBUG_VALUE: write_output_to_file:output <- $x19
	.loc	1 65 18 prologue_end            ; main.c:65:18
Lloh11:
	adrp	x1, l_.str.5@PAGE
Lloh12:
	add	x1, x1, l_.str.5@PAGEOFF
	bl	_fopen
Ltmp25:
	;DEBUG_VALUE: write_output_to_file:filename <- [DW_OP_LLVM_entry_value 1] $x0
	;DEBUG_VALUE: write_output_to_file:file <- $x0
	.loc	1 66 9                          ; main.c:66:9
	cbz	x0, LBB1_5
Ltmp26:
; %bb.1:
	;DEBUG_VALUE: write_output_to_file:file <- $x0
	;DEBUG_VALUE: write_output_to_file:filename <- [DW_OP_LLVM_entry_value 1] $x0
	;DEBUG_VALUE: write_output_to_file:output <- $x19
	;DEBUG_VALUE: write_output_to_file:size <- $w21
	.loc	1 0 0 is_stmt 0                 ; main.c:0:0
	mov	x20, x0
Ltmp27:
	;DEBUG_VALUE: write_output_to_file:file <- $x20
	;DEBUG_VALUE: i <- 0
	.loc	1 70 5 is_stmt 1                ; main.c:70:5
	cmp	w21, #1
	b.lt	LBB1_4
Ltmp28:
; %bb.2:
	;DEBUG_VALUE: i <- 0
	;DEBUG_VALUE: write_output_to_file:file <- $x20
	;DEBUG_VALUE: write_output_to_file:filename <- [DW_OP_LLVM_entry_value 1] $x0
	;DEBUG_VALUE: write_output_to_file:output <- $x19
	;DEBUG_VALUE: write_output_to_file:size <- $w21
	.loc	1 70 23 is_stmt 0               ; main.c:70:23
	mov	w22, w21
Ltmp29:
	;DEBUG_VALUE: write_output_to_file:size <- $w22
	.loc	1 0 23                          ; main.c:0:23
Lloh13:
	adrp	x21, l_.str.7@PAGE
Lloh14:
	add	x21, x21, l_.str.7@PAGEOFF
Ltmp30:
LBB1_3:                                 ; =>This Inner Loop Header: Depth=1
	;DEBUG_VALUE: write_output_to_file:file <- $x20
	;DEBUG_VALUE: write_output_to_file:filename <- [DW_OP_LLVM_entry_value 1] $x0
	;DEBUG_VALUE: i <- [DW_OP_LLVM_arg 0, DW_OP_LLVM_arg 0, DW_OP_minus, DW_OP_stack_value] undef
	.loc	1 71 31 is_stmt 1               ; main.c:71:31
	ldrsb	x8, [x19], #1
Ltmp31:
	;DEBUG_VALUE: write_output_to_file:output <- [DW_OP_LLVM_entry_value 1] $x1
	.loc	1 71 9 is_stmt 0                ; main.c:71:9
	str	x8, [sp]
	mov	x0, x20
	mov	x1, x21
	bl	_fprintf
Ltmp32:
	;DEBUG_VALUE: i <- [DW_OP_LLVM_arg 0, DW_OP_LLVM_arg 0, DW_OP_minus, DW_OP_consts 1, DW_OP_plus, DW_OP_stack_value] undef
	.loc	1 70 23 is_stmt 1               ; main.c:70:23
	subs	x22, x22, #1
Ltmp33:
	.loc	1 70 5 is_stmt 0                ; main.c:70:5
	b.ne	LBB1_3
Ltmp34:
LBB1_4:
	;DEBUG_VALUE: write_output_to_file:file <- $x20
	;DEBUG_VALUE: write_output_to_file:filename <- [DW_OP_LLVM_entry_value 1] $x0
	.loc	1 73 5 is_stmt 1                ; main.c:73:5
	mov	x0, x20
	.loc	1 73 5 epilogue_begin is_stmt 0 ; main.c:73:5
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #32]             ; 16-byte Folded Reload
Ltmp35:
	ldp	x22, x21, [sp, #16]             ; 16-byte Folded Reload
	add	sp, sp, #64
Ltmp36:
	b	_fclose
Ltmp37:
LBB1_5:
	;DEBUG_VALUE: write_output_to_file:file <- $x0
	;DEBUG_VALUE: write_output_to_file:filename <- [DW_OP_LLVM_entry_value 1] $x0
	;DEBUG_VALUE: write_output_to_file:output <- $x19
	;DEBUG_VALUE: write_output_to_file:size <- $w21
	.loc	1 67 9 is_stmt 1                ; main.c:67:9
	bl	_write_output_to_file.cold.1
Ltmp38:
	.loh AdrpAdd	Lloh11, Lloh12
	.loh AdrpAdd	Lloh13, Lloh14
Lfunc_end1:
	.cfi_endproc
                                        ; -- End function
	.globl	_main                           ; -- Begin function main
	.p2align	2
_main:                                  ; @main
Lfunc_begin2:
	.loc	1 77 0                          ; main.c:77:0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #80
	stp	x24, x23, [sp, #16]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #32]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #48]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #64]             ; 16-byte Folded Spill
	add	x29, sp, #64
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
Ltmp39:
	;DEBUG_VALUE: main:input_size <- 4194304
	;DEBUG_VALUE: main:output_size <- 8388608
	mov	w21, #4194304                   ; =0x400000
Ltmp40:
	.loc	1 81 21 prologue_end            ; main.c:81:21
	mov	w0, #4194304                    ; =0x400000
	bl	_malloc
Ltmp41:
	mov	x19, x0
Ltmp42:
	;DEBUG_VALUE: main:input <- $x19
	.loc	1 82 22                         ; main.c:82:22
	mov	w0, #8388608                    ; =0x800000
	bl	_malloc
Ltmp43:
	mov	x20, x0
Ltmp44:
	;DEBUG_VALUE: main:output <- $x20
	.loc	1 85 5                          ; main.c:85:5
	str	x21, [sp]
Lloh15:
	adrp	x0, l_.str.8@PAGE
Lloh16:
	add	x0, x0, l_.str.8@PAGEOFF
	bl	_printf
Ltmp45:
	.loc	1 89 16                         ; main.c:89:16
	cbz	x19, LBB2_6
Ltmp46:
; %bb.1:
	;DEBUG_VALUE: main:output <- $x20
	;DEBUG_VALUE: main:input <- $x19
	;DEBUG_VALUE: main:output_size <- 8388608
	;DEBUG_VALUE: main:input_size <- 4194304
	cbz	x20, LBB2_6
Ltmp47:
; %bb.2:
	;DEBUG_VALUE: main:output <- $x20
	;DEBUG_VALUE: main:input <- $x19
	;DEBUG_VALUE: main:output_size <- 8388608
	;DEBUG_VALUE: main:input_size <- 4194304
	.loc	1 94 5                          ; main.c:94:5
Lloh17:
	adrp	x0, l_.str.10@PAGE
Lloh18:
	add	x0, x0, l_.str.10@PAGEOFF
	mov	x1, x19
	mov	w2, #4194304                    ; =0x400000
	bl	_read_input_from_file
Ltmp48:
	.loc	1 96 5                          ; main.c:96:5
Lloh19:
	adrp	x0, l_str@PAGE
Lloh20:
	add	x0, x0, l_str@PAGEOFF
	bl	_puts
Ltmp49:
	;DEBUG_VALUE: i <- 0
	.loc	1 98 23                         ; main.c:98:23
	ldrsb	x8, [x19]
	.loc	1 98 9 is_stmt 0                ; main.c:98:9
	str	x8, [sp]
Lloh21:
	adrp	x21, l_.str.12@PAGE
Lloh22:
	add	x21, x21, l_.str.12@PAGEOFF
	mov	x0, x21
	bl	_printf
Ltmp50:
	;DEBUG_VALUE: i <- 1
	.loc	1 98 23                         ; main.c:98:23
	ldrsb	x8, [x19, #1]
	.loc	1 98 9                          ; main.c:98:9
	str	x8, [sp]
	mov	x0, x21
	bl	_printf
Ltmp51:
	;DEBUG_VALUE: i <- 2
	.loc	1 98 23                         ; main.c:98:23
	ldrsb	x8, [x19, #2]
	.loc	1 98 9                          ; main.c:98:9
	str	x8, [sp]
	mov	x0, x21
	bl	_printf
Ltmp52:
	;DEBUG_VALUE: i <- 3
	.loc	1 98 23                         ; main.c:98:23
	ldrsb	x8, [x19, #3]
	.loc	1 98 9                          ; main.c:98:9
	str	x8, [sp]
	mov	x0, x21
	bl	_printf
Ltmp53:
	;DEBUG_VALUE: i <- 4
	.loc	1 98 23                         ; main.c:98:23
	ldrsb	x8, [x19, #4]
	.loc	1 98 9                          ; main.c:98:9
	str	x8, [sp]
	mov	x0, x21
	bl	_printf
Ltmp54:
	;DEBUG_VALUE: i <- 5
	.loc	1 98 23                         ; main.c:98:23
	ldrsb	x8, [x19, #5]
	.loc	1 98 9                          ; main.c:98:9
	str	x8, [sp]
	mov	x0, x21
	bl	_printf
Ltmp55:
	;DEBUG_VALUE: i <- 6
	.loc	1 98 23                         ; main.c:98:23
	ldrsb	x8, [x19, #6]
	.loc	1 98 9                          ; main.c:98:9
	str	x8, [sp]
	mov	x0, x21
	bl	_printf
Ltmp56:
	;DEBUG_VALUE: i <- 7
	.loc	1 98 23                         ; main.c:98:23
	ldrsb	x8, [x19, #7]
	.loc	1 98 9                          ; main.c:98:9
	str	x8, [sp]
	mov	x0, x21
	bl	_printf
Ltmp57:
	;DEBUG_VALUE: i <- 8
	.loc	1 98 23                         ; main.c:98:23
	ldrsb	x8, [x19, #8]
	.loc	1 98 9                          ; main.c:98:9
	str	x8, [sp]
	mov	x0, x21
	bl	_printf
Ltmp58:
	;DEBUG_VALUE: i <- 9
	.loc	1 98 23                         ; main.c:98:23
	ldrsb	x8, [x19, #9]
	.loc	1 98 9                          ; main.c:98:9
	str	x8, [sp]
	mov	x0, x21
	bl	_printf
Ltmp59:
	;DEBUG_VALUE: i <- 10
	.loc	1 100 5 is_stmt 1               ; main.c:100:5
	mov	w0, #10                         ; =0xa
	bl	_putchar
Ltmp60:
	.loc	1 102 5                         ; main.c:102:5
	bl	_initialize_weights
Ltmp61:
	.loc	1 103 5                         ; main.c:103:5
	bl	_initialize_exp_lut
Ltmp62:
	.loc	1 105 5                         ; main.c:105:5
	mov	x0, x19
	mov	x1, x20
	bl	_forward
Ltmp63:
	;DEBUG_VALUE: write_output_to_file:size <- 8388608
	;DEBUG_VALUE: write_output_to_file:output <- $x20
	.loc	1 65 18                         ; main.c:65:18
Lloh23:
	adrp	x0, l_.str.14@PAGE
Lloh24:
	add	x0, x0, l_.str.14@PAGEOFF
Ltmp64:
	;DEBUG_VALUE: write_output_to_file:filename <- $x0
Lloh25:
	adrp	x1, l_.str.5@PAGE
Lloh26:
	add	x1, x1, l_.str.5@PAGEOFF
	bl	_fopen
Ltmp65:
	;DEBUG_VALUE: write_output_to_file:file <- $x0
	.loc	1 66 9                          ; main.c:66:9
	cbz	x0, LBB2_7
Ltmp66:
; %bb.3:
	;DEBUG_VALUE: write_output_to_file:file <- $x0
	;DEBUG_VALUE: write_output_to_file:output <- $x20
	;DEBUG_VALUE: write_output_to_file:size <- 8388608
	;DEBUG_VALUE: i <- 10
	;DEBUG_VALUE: main:output <- $x20
	;DEBUG_VALUE: main:input <- $x19
	;DEBUG_VALUE: main:output_size <- 8388608
	;DEBUG_VALUE: main:input_size <- 4194304
	.loc	1 0 9 is_stmt 0                 ; main.c:0:9
	mov	x21, x0
Ltmp67:
	;DEBUG_VALUE: write_output_to_file:file <- $x21
	mov	x23, #0                         ; =0x0
Lloh27:
	adrp	x22, l_.str.7@PAGE
Lloh28:
	add	x22, x22, l_.str.7@PAGEOFF
Ltmp68:
LBB2_4:                                 ; =>This Inner Loop Header: Depth=1
	;DEBUG_VALUE: write_output_to_file:file <- $x21
	;DEBUG_VALUE: write_output_to_file:output <- $x20
	;DEBUG_VALUE: write_output_to_file:size <- 8388608
	;DEBUG_VALUE: main:output <- $x20
	;DEBUG_VALUE: main:input <- $x19
	;DEBUG_VALUE: main:output_size <- 8388608
	;DEBUG_VALUE: main:input_size <- 4194304
	;DEBUG_VALUE: i <- $x23
	.loc	1 71 31 is_stmt 1               ; main.c:71:31
	ldrsb	x8, [x20, x23]
	.loc	1 71 9 is_stmt 0                ; main.c:71:9
	str	x8, [sp]
	mov	x0, x21
	mov	x1, x22
	bl	_fprintf
Ltmp69:
	.loc	1 70 32 is_stmt 1               ; main.c:70:32
	add	x23, x23, #1
Ltmp70:
	;DEBUG_VALUE: i <- $x23
	.loc	1 70 5 is_stmt 0                ; main.c:70:5
	cmp	x23, #2048, lsl #12             ; =8388608
	b.ne	LBB2_4
Ltmp71:
; %bb.5:
	;DEBUG_VALUE: write_output_to_file:file <- $x21
	;DEBUG_VALUE: write_output_to_file:output <- $x20
	;DEBUG_VALUE: write_output_to_file:size <- 8388608
	;DEBUG_VALUE: main:output <- $x20
	;DEBUG_VALUE: main:input <- $x19
	;DEBUG_VALUE: main:output_size <- 8388608
	;DEBUG_VALUE: main:input_size <- 4194304
	.loc	1 73 5 is_stmt 1                ; main.c:73:5
	mov	x0, x21
	bl	_fclose
Ltmp72:
	.loc	1 115 5                         ; main.c:115:5
	mov	x0, x19
	bl	_free
Ltmp73:
	.loc	1 116 5                         ; main.c:116:5
	mov	x0, x20
	bl	_free
Ltmp74:
	.loc	1 0 5 is_stmt 0                 ; main.c:0:5
	mov	w0, #0                          ; =0x0
	.loc	1 120 1 epilogue_begin is_stmt 1 ; main.c:120:1
	ldp	x29, x30, [sp, #64]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #48]             ; 16-byte Folded Reload
Ltmp75:
	ldp	x22, x21, [sp, #32]             ; 16-byte Folded Reload
Ltmp76:
	ldp	x24, x23, [sp, #16]             ; 16-byte Folded Reload
	add	sp, sp, #80
	ret
Ltmp77:
LBB2_6:
	;DEBUG_VALUE: main:output <- $x20
	;DEBUG_VALUE: main:input <- $x19
	;DEBUG_VALUE: main:output_size <- 8388608
	;DEBUG_VALUE: main:input_size <- 4194304
	.loc	1 90 9                          ; main.c:90:9
Lloh29:
	adrp	x0, l_.str.9@PAGE
Lloh30:
	add	x0, x0, l_.str.9@PAGEOFF
	bl	_perror
Ltmp78:
	.loc	1 0 9 is_stmt 0                 ; main.c:0:9
	mov	w0, #1                          ; =0x1
Ltmp79:
	.loc	1 120 1 epilogue_begin is_stmt 1 ; main.c:120:1
	ldp	x29, x30, [sp, #64]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #48]             ; 16-byte Folded Reload
Ltmp80:
	ldp	x22, x21, [sp, #32]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #16]             ; 16-byte Folded Reload
	add	sp, sp, #80
	ret
Ltmp81:
LBB2_7:
	;DEBUG_VALUE: write_output_to_file:file <- $x0
	;DEBUG_VALUE: write_output_to_file:output <- $x20
	;DEBUG_VALUE: write_output_to_file:size <- 8388608
	;DEBUG_VALUE: main:output <- $x20
	;DEBUG_VALUE: main:input <- $x19
	;DEBUG_VALUE: main:output_size <- 8388608
	;DEBUG_VALUE: main:input_size <- 4194304
	.loc	1 67 9                          ; main.c:67:9
	bl	_main.cold.1
Ltmp82:
	.loh AdrpAdd	Lloh15, Lloh16
	.loh AdrpAdd	Lloh25, Lloh26
	.loh AdrpAdd	Lloh23, Lloh24
	.loh AdrpAdd	Lloh21, Lloh22
	.loh AdrpAdd	Lloh19, Lloh20
	.loh AdrpAdd	Lloh17, Lloh18
	.loh AdrpAdd	Lloh27, Lloh28
	.loh AdrpAdd	Lloh29, Lloh30
Lfunc_end2:
	.cfi_endproc
	.file	7 "/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk/usr/include/sys/_types" "_int8_t.h" md5 0x5052dfe782b50fed2355b8f61a9e4fa9
	.file	8 "/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk/usr/include/malloc" "_malloc.h" md5 0x169c9ed8662627a4a4d92e321cfa866c
	.file	9 "/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk/usr/include/sys/_types" "_size_t.h" md5 0xf7981334d28e0c246f35cd24042aa2a4
	.file	10 "/Users/lmwgsrwny/Documents/_E8/C_Lang_Mamba/src" "forward.h" md5 0x81eea202656b8bc528af10b6a2eb64cc
                                        ; -- End function
	.p2align	2                               ; -- Begin function read_input_from_file.cold.1
_read_input_from_file.cold.1:           ; @read_input_from_file.cold.1
Lfunc_begin3:
	.loc	1 0 0                           ; main.c:0:0
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Ltmp83:
	.loc	1 47 9 prologue_end             ; main.c:47:9
Lloh31:
	adrp	x0, l_.str.1@PAGE
Lloh32:
	add	x0, x0, l_.str.1@PAGEOFF
	bl	_perror
	bl	_OUTLINED_FUNCTION_0
Ltmp84:
	.loh AdrpAdd	Lloh31, Lloh32
Lfunc_end3:
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function write_output_to_file.cold.1
_write_output_to_file.cold.1:           ; @write_output_to_file.cold.1
Lfunc_begin4:
	.loc	1 0 0                           ; main.c:0:0
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Ltmp85:
	.loc	1 67 9 prologue_end             ; main.c:67:9
Lloh33:
	adrp	x0, l_.str.6@PAGE
Lloh34:
	add	x0, x0, l_.str.6@PAGEOFF
	bl	_perror
	bl	_OUTLINED_FUNCTION_0
Ltmp86:
	.loh AdrpAdd	Lloh33, Lloh34
Lfunc_end4:
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function main.cold.1
_main.cold.1:                           ; @main.cold.1
Lfunc_begin5:
	.loc	1 0 0                           ; main.c:0:0
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Ltmp87:
	.loc	1 67 9 prologue_end             ; main.c:67:9
Lloh35:
	adrp	x0, l_.str.6@PAGE
Lloh36:
	add	x0, x0, l_.str.6@PAGEOFF
	bl	_perror
	bl	_OUTLINED_FUNCTION_0
Ltmp88:
	.loh AdrpAdd	Lloh35, Lloh36
Lfunc_end5:
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function OUTLINED_FUNCTION_0
_OUTLINED_FUNCTION_0:                   ; @OUTLINED_FUNCTION_0 Thunk
Lfunc_begin6:
	.cfi_startproc
; %bb.0:
	mov	w0, #1                          ; =0x1
	b	_exit
Lfunc_end6:
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.asciz	"r"

l_.str.1:                               ; @.str.1
	.asciz	"Failed to open file"

l_.str.2:                               ; @.str.2
	.asciz	"%hhd"

l_.str.3:                               ; @.str.3
	.asciz	"Error reading input at index %d\n"

l_.str.4:                               ; @.str.4
	.asciz	"Input successfully loaded from file: %s\n"

l_.str.5:                               ; @.str.5
	.asciz	"w"

l_.str.6:                               ; @.str.6
	.asciz	"Failed to open output file"

l_.str.7:                               ; @.str.7
	.asciz	"%d\n"

l_.str.8:                               ; @.str.8
	.asciz	"Input size: %d\n"

l_.str.9:                               ; @.str.9
	.asciz	"Failed to allocate memory"

l_.str.10:                              ; @.str.10
	.asciz	"data/input.txt"

l_.str.12:                              ; @.str.12
	.asciz	"%d "

l_.str.14:                              ; @.str.14
	.asciz	"data/output_c.txt"

l_str:                                  ; @str
	.asciz	"First few inputs:"

	.section	__DWARF,__debug_loclists,regular,debug
Lsection_debug_loc0:
.set Lset0, Ldebug_list_header_end0-Ldebug_list_header_start0 ; Length
	.long	Lset0
Ldebug_list_header_start0:
	.short	5                               ; Version
	.byte	8                               ; Address size
	.byte	0                               ; Segment selector size
	.long	17                              ; Offset entry count
Lloclists_table_base0:
.set Lset1, Ldebug_loc0-Lloclists_table_base0
	.long	Lset1
.set Lset2, Ldebug_loc1-Lloclists_table_base0
	.long	Lset2
.set Lset3, Ldebug_loc2-Lloclists_table_base0
	.long	Lset3
.set Lset4, Ldebug_loc3-Lloclists_table_base0
	.long	Lset4
.set Lset5, Ldebug_loc4-Lloclists_table_base0
	.long	Lset5
.set Lset6, Ldebug_loc5-Lloclists_table_base0
	.long	Lset6
.set Lset7, Ldebug_loc6-Lloclists_table_base0
	.long	Lset7
.set Lset8, Ldebug_loc7-Lloclists_table_base0
	.long	Lset8
.set Lset9, Ldebug_loc8-Lloclists_table_base0
	.long	Lset9
.set Lset10, Ldebug_loc9-Lloclists_table_base0
	.long	Lset10
.set Lset11, Ldebug_loc10-Lloclists_table_base0
	.long	Lset11
.set Lset12, Ldebug_loc11-Lloclists_table_base0
	.long	Lset12
.set Lset13, Ldebug_loc12-Lloclists_table_base0
	.long	Lset13
.set Lset14, Ldebug_loc13-Lloclists_table_base0
	.long	Lset14
.set Lset15, Ldebug_loc14-Lloclists_table_base0
	.long	Lset15
.set Lset16, Ldebug_loc15-Lloclists_table_base0
	.long	Lset16
.set Lset17, Ldebug_loc16-Lloclists_table_base0
	.long	Lset17
Ldebug_loc0:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Lfunc_begin0-Lfunc_begin0      ;   starting offset
	.uleb128 Ltmp2-Lfunc_begin0             ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	80                              ; DW_OP_reg0
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp2-Lfunc_begin0             ;   starting offset
	.uleb128 Ltmp16-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	100                             ; DW_OP_reg20
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp16-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp18-Lfunc_begin0            ;   ending offset
	.byte	4                               ; Loc expr size
	.byte	163                             ; DW_OP_entry_value
	.byte	1                               ; 1
	.byte	80                              ; DW_OP_reg0
	.byte	159                             ; DW_OP_stack_value
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp18-Lfunc_begin0            ;   starting offset
	.uleb128 Lfunc_end0-Lfunc_begin0        ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	100                             ; DW_OP_reg20
	.byte	0                               ; DW_LLE_end_of_list
Ldebug_loc1:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Lfunc_begin0-Lfunc_begin0      ;   starting offset
	.uleb128 Ltmp1-Lfunc_begin0             ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	81                              ; DW_OP_reg1
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp1-Lfunc_begin0             ;   starting offset
	.uleb128 Ltmp17-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	101                             ; DW_OP_reg21
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp17-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp18-Lfunc_begin0            ;   ending offset
	.byte	4                               ; Loc expr size
	.byte	163                             ; DW_OP_entry_value
	.byte	1                               ; 1
	.byte	81                              ; DW_OP_reg1
	.byte	159                             ; DW_OP_stack_value
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp18-Lfunc_begin0            ;   starting offset
	.uleb128 Lfunc_end0-Lfunc_begin0        ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	101                             ; DW_OP_reg21
	.byte	0                               ; DW_LLE_end_of_list
Ldebug_loc2:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Lfunc_begin0-Lfunc_begin0      ;   starting offset
	.uleb128 Ltmp0-Lfunc_begin0             ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	82                              ; DW_OP_reg2
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp0-Lfunc_begin0             ;   starting offset
	.uleb128 Ltmp8-Lfunc_begin0             ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	102                             ; DW_OP_reg22
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp8-Lfunc_begin0             ;   starting offset
	.uleb128 Ltmp13-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	104                             ; DW_OP_reg24
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp18-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp21-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	104                             ; DW_OP_reg24
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp21-Lfunc_begin0            ;   starting offset
	.uleb128 Lfunc_end0-Lfunc_begin0        ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	102                             ; DW_OP_reg22
	.byte	0                               ; DW_LLE_end_of_list
Ldebug_loc3:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp3-Lfunc_begin0             ;   starting offset
	.uleb128 Ltmp5-Lfunc_begin0             ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	80                              ; DW_OP_reg0
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp5-Lfunc_begin0             ;   starting offset
	.uleb128 Ltmp16-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	99                              ; DW_OP_reg19
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp18-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp21-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	99                              ; DW_OP_reg19
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp21-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp22-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	80                              ; DW_OP_reg0
	.byte	0                               ; DW_LLE_end_of_list
Ldebug_loc4:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp5-Lfunc_begin0             ;   starting offset
	.uleb128 Ltmp9-Lfunc_begin0             ;   ending offset
	.byte	3                               ; Loc expr size
	.byte	17                              ; DW_OP_consts
	.byte	0                               ; 0
	.byte	159                             ; DW_OP_stack_value
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp9-Lfunc_begin0             ;   starting offset
	.uleb128 Ltmp13-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	103                             ; DW_OP_reg23
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp18-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp21-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	103                             ; DW_OP_reg23
	.byte	0                               ; DW_LLE_end_of_list
Ldebug_loc5:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Lfunc_begin1-Lfunc_begin0      ;   starting offset
	.uleb128 Ltmp25-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	80                              ; DW_OP_reg0
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp25-Lfunc_begin0            ;   starting offset
	.uleb128 Lfunc_end1-Lfunc_begin0        ;   ending offset
	.byte	4                               ; Loc expr size
	.byte	163                             ; DW_OP_entry_value
	.byte	1                               ; 1
	.byte	80                              ; DW_OP_reg0
	.byte	159                             ; DW_OP_stack_value
	.byte	0                               ; DW_LLE_end_of_list
Ldebug_loc6:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Lfunc_begin1-Lfunc_begin0      ;   starting offset
	.uleb128 Ltmp24-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	81                              ; DW_OP_reg1
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp24-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp30-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	99                              ; DW_OP_reg19
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp31-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp34-Lfunc_begin0            ;   ending offset
	.byte	4                               ; Loc expr size
	.byte	163                             ; DW_OP_entry_value
	.byte	1                               ; 1
	.byte	81                              ; DW_OP_reg1
	.byte	159                             ; DW_OP_stack_value
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp37-Lfunc_begin0            ;   starting offset
	.uleb128 Lfunc_end1-Lfunc_begin0        ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	99                              ; DW_OP_reg19
	.byte	0                               ; DW_LLE_end_of_list
Ldebug_loc7:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Lfunc_begin1-Lfunc_begin0      ;   starting offset
	.uleb128 Ltmp23-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	82                              ; DW_OP_reg2
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp23-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp29-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	101                             ; DW_OP_reg21
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp29-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp30-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	102                             ; DW_OP_reg22
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp37-Lfunc_begin0            ;   starting offset
	.uleb128 Lfunc_end1-Lfunc_begin0        ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	101                             ; DW_OP_reg21
	.byte	0                               ; DW_LLE_end_of_list
Ldebug_loc8:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp25-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp27-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	80                              ; DW_OP_reg0
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp27-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp35-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	100                             ; DW_OP_reg20
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp37-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp38-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	80                              ; DW_OP_reg0
	.byte	0                               ; DW_LLE_end_of_list
Ldebug_loc9:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp27-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp30-Lfunc_begin0            ;   ending offset
	.byte	3                               ; Loc expr size
	.byte	17                              ; DW_OP_consts
	.byte	0                               ; 0
	.byte	159                             ; DW_OP_stack_value
	.byte	0                               ; DW_LLE_end_of_list
Ldebug_loc10:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp42-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp75-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	99                              ; DW_OP_reg19
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp77-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp80-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	99                              ; DW_OP_reg19
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp81-Lfunc_begin0            ;   starting offset
	.uleb128 Lfunc_end2-Lfunc_begin0        ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	99                              ; DW_OP_reg19
	.byte	0                               ; DW_LLE_end_of_list
Ldebug_loc11:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp44-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp75-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	100                             ; DW_OP_reg20
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp77-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp80-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	100                             ; DW_OP_reg20
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp81-Lfunc_begin0            ;   starting offset
	.uleb128 Lfunc_end2-Lfunc_begin0        ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	100                             ; DW_OP_reg20
	.byte	0                               ; DW_LLE_end_of_list
Ldebug_loc12:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp49-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp50-Lfunc_begin0            ;   ending offset
	.byte	3                               ; Loc expr size
	.byte	17                              ; DW_OP_consts
	.byte	0                               ; 0
	.byte	159                             ; DW_OP_stack_value
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp50-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp51-Lfunc_begin0            ;   ending offset
	.byte	3                               ; Loc expr size
	.byte	17                              ; DW_OP_consts
	.byte	1                               ; 1
	.byte	159                             ; DW_OP_stack_value
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp51-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp52-Lfunc_begin0            ;   ending offset
	.byte	3                               ; Loc expr size
	.byte	17                              ; DW_OP_consts
	.byte	2                               ; 2
	.byte	159                             ; DW_OP_stack_value
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp52-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp53-Lfunc_begin0            ;   ending offset
	.byte	3                               ; Loc expr size
	.byte	17                              ; DW_OP_consts
	.byte	3                               ; 3
	.byte	159                             ; DW_OP_stack_value
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp53-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp54-Lfunc_begin0            ;   ending offset
	.byte	3                               ; Loc expr size
	.byte	17                              ; DW_OP_consts
	.byte	4                               ; 4
	.byte	159                             ; DW_OP_stack_value
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp54-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp55-Lfunc_begin0            ;   ending offset
	.byte	3                               ; Loc expr size
	.byte	17                              ; DW_OP_consts
	.byte	5                               ; 5
	.byte	159                             ; DW_OP_stack_value
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp55-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp56-Lfunc_begin0            ;   ending offset
	.byte	3                               ; Loc expr size
	.byte	17                              ; DW_OP_consts
	.byte	6                               ; 6
	.byte	159                             ; DW_OP_stack_value
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp56-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp57-Lfunc_begin0            ;   ending offset
	.byte	3                               ; Loc expr size
	.byte	17                              ; DW_OP_consts
	.byte	7                               ; 7
	.byte	159                             ; DW_OP_stack_value
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp57-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp58-Lfunc_begin0            ;   ending offset
	.byte	3                               ; Loc expr size
	.byte	17                              ; DW_OP_consts
	.byte	8                               ; 8
	.byte	159                             ; DW_OP_stack_value
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp58-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp59-Lfunc_begin0            ;   ending offset
	.byte	3                               ; Loc expr size
	.byte	17                              ; DW_OP_consts
	.byte	9                               ; 9
	.byte	159                             ; DW_OP_stack_value
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp59-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp66-Lfunc_begin0            ;   ending offset
	.byte	3                               ; Loc expr size
	.byte	17                              ; DW_OP_consts
	.byte	10                              ; 10
	.byte	159                             ; DW_OP_stack_value
	.byte	0                               ; DW_LLE_end_of_list
Ldebug_loc13:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp63-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp77-Lfunc_begin0            ;   ending offset
	.byte	6                               ; Loc expr size
	.byte	17                              ; DW_OP_consts
	.byte	128                             ; 8388608
	.byte	128                             ; 
	.byte	128                             ; 
	.byte	4                               ; 
	.byte	159                             ; DW_OP_stack_value
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp81-Lfunc_begin0            ;   starting offset
	.uleb128 Lfunc_end2-Lfunc_begin0        ;   ending offset
	.byte	6                               ; Loc expr size
	.byte	17                              ; DW_OP_consts
	.byte	128                             ; 8388608
	.byte	128                             ; 
	.byte	128                             ; 
	.byte	4                               ; 
	.byte	159                             ; DW_OP_stack_value
	.byte	0                               ; DW_LLE_end_of_list
Ldebug_loc14:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp63-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp75-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	100                             ; DW_OP_reg20
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp81-Lfunc_begin0            ;   starting offset
	.uleb128 Lfunc_end2-Lfunc_begin0        ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	100                             ; DW_OP_reg20
	.byte	0                               ; DW_LLE_end_of_list
Ldebug_loc15:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp64-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp65-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	80                              ; DW_OP_reg0
	.byte	0                               ; DW_LLE_end_of_list
Ldebug_loc16:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp65-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp67-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	80                              ; DW_OP_reg0
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp67-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp76-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	101                             ; DW_OP_reg21
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp81-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp82-Lfunc_begin0            ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	80                              ; DW_OP_reg0
	.byte	0                               ; DW_LLE_end_of_list
Ldebug_list_header_end0:
	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	37                              ; DW_FORM_strx1
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.ascii	"\202|"                         ; DW_AT_LLVM_sysroot
	.byte	37                              ; DW_FORM_strx1
	.ascii	"\357\177"                      ; DW_AT_APPLE_sdk
	.byte	37                              ; DW_FORM_strx1
	.byte	114                             ; DW_AT_str_offsets_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	37                              ; DW_FORM_strx1
	.ascii	"\341\177"                      ; DW_AT_APPLE_optimized
	.byte	25                              ; DW_FORM_flag_present
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	115                             ; DW_AT_addr_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	116                             ; DW_AT_rnglists_base
	.byte	23                              ; DW_FORM_sec_offset
	.ascii	"\214\001"                      ; DW_AT_loclists_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	52                              ; DW_TAG_variable
	.byte	0                               ; DW_CHILDREN_no
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	2                               ; DW_AT_location
	.byte	24                              ; DW_FORM_exprloc
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	1                               ; DW_TAG_array_type
	.byte	1                               ; DW_CHILDREN_yes
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	33                              ; DW_TAG_subrange_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	55                              ; DW_AT_count
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
	.byte	36                              ; DW_TAG_base_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	62                              ; DW_AT_encoding
	.byte	11                              ; DW_FORM_data1
	.byte	11                              ; DW_AT_byte_size
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	6                               ; Abbreviation Code
	.byte	36                              ; DW_TAG_base_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	11                              ; DW_AT_byte_size
	.byte	11                              ; DW_FORM_data1
	.byte	62                              ; DW_AT_encoding
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	7                               ; Abbreviation Code
	.byte	52                              ; DW_TAG_variable
	.byte	0                               ; DW_CHILDREN_no
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	8                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	64                              ; DW_AT_frame_base
	.byte	24                              ; DW_FORM_exprloc
	.byte	122                             ; DW_AT_call_all_calls
	.byte	25                              ; DW_FORM_flag_present
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	39                              ; DW_AT_prototyped
	.byte	25                              ; DW_FORM_flag_present
	.byte	63                              ; DW_AT_external
	.byte	25                              ; DW_FORM_flag_present
	.ascii	"\341\177"                      ; DW_AT_APPLE_optimized
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	9                               ; Abbreviation Code
	.byte	5                               ; DW_TAG_formal_parameter
	.byte	0                               ; DW_CHILDREN_no
	.byte	2                               ; DW_AT_location
	.byte	34                              ; DW_FORM_loclistx
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	10                              ; Abbreviation Code
	.byte	52                              ; DW_TAG_variable
	.byte	0                               ; DW_CHILDREN_no
	.byte	2                               ; DW_AT_location
	.byte	34                              ; DW_FORM_loclistx
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	11                              ; Abbreviation Code
	.byte	11                              ; DW_TAG_lexical_block
	.byte	1                               ; DW_CHILDREN_yes
	.byte	85                              ; DW_AT_ranges
	.byte	35                              ; DW_FORM_rnglistx
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	12                              ; Abbreviation Code
	.byte	72                              ; DW_TAG_call_site
	.byte	1                               ; DW_CHILDREN_yes
	.byte	127                             ; DW_AT_call_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	125                             ; DW_AT_call_return_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	13                              ; Abbreviation Code
	.byte	73                              ; DW_TAG_call_site_parameter
	.byte	0                               ; DW_CHILDREN_no
	.byte	2                               ; DW_AT_location
	.byte	24                              ; DW_FORM_exprloc
	.byte	126                             ; DW_AT_call_value
	.byte	24                              ; DW_FORM_exprloc
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	14                              ; Abbreviation Code
	.byte	72                              ; DW_TAG_call_site
	.byte	0                               ; DW_CHILDREN_no
	.byte	127                             ; DW_AT_call_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	125                             ; DW_AT_call_return_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	15                              ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	110                             ; DW_AT_linkage_name
	.byte	37                              ; DW_FORM_strx1
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	39                              ; DW_AT_prototyped
	.byte	25                              ; DW_FORM_flag_present
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	60                              ; DW_AT_declaration
	.byte	25                              ; DW_FORM_flag_present
	.byte	63                              ; DW_AT_external
	.byte	25                              ; DW_FORM_flag_present
	.ascii	"\341\177"                      ; DW_AT_APPLE_optimized
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	16                              ; Abbreviation Code
	.byte	5                               ; DW_TAG_formal_parameter
	.byte	0                               ; DW_CHILDREN_no
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	17                              ; Abbreviation Code
	.byte	15                              ; DW_TAG_pointer_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	18                              ; Abbreviation Code
	.byte	22                              ; DW_TAG_typedef
	.byte	0                               ; DW_CHILDREN_no
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	19                              ; Abbreviation Code
	.byte	19                              ; DW_TAG_structure_type
	.byte	1                               ; DW_CHILDREN_yes
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	11                              ; DW_AT_byte_size
	.byte	11                              ; DW_FORM_data1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	20                              ; Abbreviation Code
	.byte	13                              ; DW_TAG_member
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	56                              ; DW_AT_data_member_location
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	21                              ; Abbreviation Code
	.byte	15                              ; DW_TAG_pointer_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	22                              ; Abbreviation Code
	.byte	21                              ; DW_TAG_subroutine_type
	.byte	1                               ; DW_CHILDREN_yes
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	39                              ; DW_AT_prototyped
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	23                              ; Abbreviation Code
	.byte	38                              ; DW_TAG_const_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	24                              ; Abbreviation Code
	.byte	19                              ; DW_TAG_structure_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	60                              ; DW_AT_declaration
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	25                              ; Abbreviation Code
	.byte	55                              ; DW_TAG_restrict_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	26                              ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	39                              ; DW_AT_prototyped
	.byte	25                              ; DW_FORM_flag_present
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	60                              ; DW_AT_declaration
	.byte	25                              ; DW_FORM_flag_present
	.byte	63                              ; DW_AT_external
	.byte	25                              ; DW_FORM_flag_present
	.ascii	"\341\177"                      ; DW_AT_APPLE_optimized
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	27                              ; Abbreviation Code
	.byte	24                              ; DW_TAG_unspecified_parameters
	.byte	0                               ; DW_CHILDREN_no
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	28                              ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	39                              ; DW_AT_prototyped
	.byte	25                              ; DW_FORM_flag_present
	.byte	60                              ; DW_AT_declaration
	.byte	25                              ; DW_FORM_flag_present
	.byte	63                              ; DW_AT_external
	.byte	25                              ; DW_FORM_flag_present
	.ascii	"\341\177"                      ; DW_AT_APPLE_optimized
	.byte	25                              ; DW_FORM_flag_present
	.ascii	"\207\001"                      ; DW_AT_noreturn
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	29                              ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	64                              ; DW_AT_frame_base
	.byte	24                              ; DW_FORM_exprloc
	.byte	110                             ; DW_AT_linkage_name
	.byte	37                              ; DW_FORM_strx1
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.ascii	"\341\177"                      ; DW_AT_APPLE_optimized
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	30                              ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	64                              ; DW_AT_frame_base
	.byte	24                              ; DW_FORM_exprloc
	.byte	122                             ; DW_AT_call_all_calls
	.byte	25                              ; DW_FORM_flag_present
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	31                              ; Abbreviation Code
	.byte	5                               ; DW_TAG_formal_parameter
	.byte	0                               ; DW_CHILDREN_no
	.byte	2                               ; DW_AT_location
	.byte	34                              ; DW_FORM_loclistx
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	32                              ; Abbreviation Code
	.byte	52                              ; DW_TAG_variable
	.byte	0                               ; DW_CHILDREN_no
	.byte	2                               ; DW_AT_location
	.byte	34                              ; DW_FORM_loclistx
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	33                              ; Abbreviation Code
	.byte	11                              ; DW_TAG_lexical_block
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	34                              ; Abbreviation Code
	.byte	72                              ; DW_TAG_call_site
	.byte	0                               ; DW_CHILDREN_no
	.byte	127                             ; DW_AT_call_origin
	.byte	19                              ; DW_FORM_ref4
	.ascii	"\202\001"                      ; DW_AT_call_tail_call
	.byte	25                              ; DW_FORM_flag_present
	.ascii	"\201\001"                      ; DW_AT_call_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	35                              ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	39                              ; DW_AT_prototyped
	.byte	25                              ; DW_FORM_flag_present
	.byte	63                              ; DW_AT_external
	.byte	25                              ; DW_FORM_flag_present
	.ascii	"\341\177"                      ; DW_AT_APPLE_optimized
	.byte	25                              ; DW_FORM_flag_present
	.byte	32                              ; DW_AT_inline
	.byte	33                              ; DW_FORM_implicit_const
	.byte	1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	36                              ; Abbreviation Code
	.byte	5                               ; DW_TAG_formal_parameter
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	37                              ; Abbreviation Code
	.byte	52                              ; DW_TAG_variable
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	38                              ; Abbreviation Code
	.byte	11                              ; DW_TAG_lexical_block
	.byte	1                               ; DW_CHILDREN_yes
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	39                              ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	64                              ; DW_AT_frame_base
	.byte	24                              ; DW_FORM_exprloc
	.byte	122                             ; DW_AT_call_all_calls
	.byte	25                              ; DW_FORM_flag_present
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	63                              ; DW_AT_external
	.byte	25                              ; DW_FORM_flag_present
	.ascii	"\341\177"                      ; DW_AT_APPLE_optimized
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	40                              ; Abbreviation Code
	.byte	52                              ; DW_TAG_variable
	.byte	0                               ; DW_CHILDREN_no
	.byte	28                              ; DW_AT_const_value
	.byte	13                              ; DW_FORM_sdata
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	41                              ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	35                              ; DW_FORM_rnglistx
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	42                              ; Abbreviation Code
	.byte	52                              ; DW_TAG_variable
	.byte	0                               ; DW_CHILDREN_no
	.byte	2                               ; DW_AT_location
	.byte	24                              ; DW_FORM_exprloc
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	43                              ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	39                              ; DW_AT_prototyped
	.byte	25                              ; DW_FORM_flag_present
	.byte	60                              ; DW_AT_declaration
	.byte	25                              ; DW_FORM_flag_present
	.byte	63                              ; DW_AT_external
	.byte	25                              ; DW_FORM_flag_present
	.ascii	"\341\177"                      ; DW_AT_APPLE_optimized
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	44                              ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	5                               ; DW_FORM_data2
	.byte	39                              ; DW_AT_prototyped
	.byte	25                              ; DW_FORM_flag_present
	.byte	60                              ; DW_AT_declaration
	.byte	25                              ; DW_FORM_flag_present
	.byte	63                              ; DW_AT_external
	.byte	25                              ; DW_FORM_flag_present
	.ascii	"\341\177"                      ; DW_AT_APPLE_optimized
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	45                              ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	64                              ; DW_AT_frame_base
	.byte	24                              ; DW_FORM_exprloc
	.byte	110                             ; DW_AT_linkage_name
	.byte	37                              ; DW_FORM_strx1
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.ascii	"\341\177"                      ; DW_AT_APPLE_optimized
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	46                              ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	47                              ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.ascii	"\347\177"                      ; DW_AT_APPLE_omit_frame_ptr
	.byte	25                              ; DW_FORM_flag_present
	.byte	64                              ; DW_AT_frame_base
	.byte	24                              ; DW_FORM_exprloc
	.byte	110                             ; DW_AT_linkage_name
	.byte	37                              ; DW_FORM_strx1
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	52                              ; DW_AT_artificial
	.byte	25                              ; DW_FORM_flag_present
	.byte	63                              ; DW_AT_external
	.byte	25                              ; DW_FORM_flag_present
	.ascii	"\341\177"                      ; DW_AT_APPLE_optimized
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
.set Lset18, Ldebug_info_end0-Ldebug_info_start0 ; Length of Unit
	.long	Lset18
Ldebug_info_start0:
	.short	5                               ; DWARF version number
	.byte	1                               ; DWARF Unit Type
	.byte	8                               ; Address Size (in bytes)
.set Lset19, Lsection_abbrev-Lsection_abbrev ; Offset Into Abbrev. Section
	.long	Lset19
	.byte	1                               ; Abbrev [1] 0xc:0x697 DW_TAG_compile_unit
	.byte	0                               ; DW_AT_producer
	.short	12                              ; DW_AT_language
	.byte	1                               ; DW_AT_name
	.byte	2                               ; DW_AT_LLVM_sysroot
	.byte	3                               ; DW_AT_APPLE_sdk
.set Lset20, Lstr_offsets_base0-Lsection_str_off ; DW_AT_str_offsets_base
	.long	Lset20
.set Lset21, Lline_table_start0-Lsection_line ; DW_AT_stmt_list
	.long	Lset21
	.byte	4                               ; DW_AT_comp_dir
                                        ; DW_AT_APPLE_optimized
	.byte	13                              ; DW_AT_low_pc
.set Lset22, Lfunc_end6-Lfunc_begin0    ; DW_AT_high_pc
	.long	Lset22
.set Lset23, Laddr_table_base0-Lsection_info0 ; DW_AT_addr_base
	.long	Lset23
.set Lset24, Lrnglists_table_base0-Ldebug_range0 ; DW_AT_rnglists_base
	.long	Lset24
.set Lset25, Lloclists_table_base0-Lsection_debug_loc0 ; DW_AT_loclists_base
	.long	Lset25
	.byte	2                               ; Abbrev [2] 0x2d:0xa DW_TAG_variable
	.long	55                              ; DW_AT_type
	.byte	1                               ; DW_AT_decl_file
	.byte	45                              ; DW_AT_decl_line
	.byte	2                               ; DW_AT_location
	.byte	161
	.byte	0
	.byte	3                               ; Abbrev [3] 0x37:0xc DW_TAG_array_type
	.long	67                              ; DW_AT_type
	.byte	4                               ; Abbrev [4] 0x3c:0x6 DW_TAG_subrange_type
	.long	71                              ; DW_AT_type
	.byte	2                               ; DW_AT_count
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x43:0x4 DW_TAG_base_type
	.byte	5                               ; DW_AT_name
	.byte	6                               ; DW_AT_encoding
	.byte	1                               ; DW_AT_byte_size
	.byte	6                               ; Abbrev [6] 0x47:0x4 DW_TAG_base_type
	.byte	6                               ; DW_AT_name
	.byte	8                               ; DW_AT_byte_size
	.byte	7                               ; DW_AT_encoding
	.byte	2                               ; Abbrev [2] 0x4b:0xa DW_TAG_variable
	.long	85                              ; DW_AT_type
	.byte	1                               ; DW_AT_decl_file
	.byte	47                              ; DW_AT_decl_line
	.byte	2                               ; DW_AT_location
	.byte	161
	.byte	1
	.byte	3                               ; Abbrev [3] 0x55:0xc DW_TAG_array_type
	.long	67                              ; DW_AT_type
	.byte	4                               ; Abbrev [4] 0x5a:0x6 DW_TAG_subrange_type
	.long	71                              ; DW_AT_type
	.byte	20                              ; DW_AT_count
	.byte	0                               ; End Of Children Mark
	.byte	2                               ; Abbrev [2] 0x61:0xa DW_TAG_variable
	.long	107                             ; DW_AT_type
	.byte	1                               ; DW_AT_decl_file
	.byte	52                              ; DW_AT_decl_line
	.byte	2                               ; DW_AT_location
	.byte	161
	.byte	2
	.byte	3                               ; Abbrev [3] 0x6b:0xc DW_TAG_array_type
	.long	67                              ; DW_AT_type
	.byte	4                               ; Abbrev [4] 0x70:0x6 DW_TAG_subrange_type
	.long	71                              ; DW_AT_type
	.byte	5                               ; DW_AT_count
	.byte	0                               ; End Of Children Mark
	.byte	2                               ; Abbrev [2] 0x77:0xa DW_TAG_variable
	.long	129                             ; DW_AT_type
	.byte	1                               ; DW_AT_decl_file
	.byte	53                              ; DW_AT_decl_line
	.byte	2                               ; DW_AT_location
	.byte	161
	.byte	3
	.byte	3                               ; Abbrev [3] 0x81:0xc DW_TAG_array_type
	.long	67                              ; DW_AT_type
	.byte	4                               ; Abbrev [4] 0x86:0x6 DW_TAG_subrange_type
	.long	71                              ; DW_AT_type
	.byte	33                              ; DW_AT_count
	.byte	0                               ; End Of Children Mark
	.byte	2                               ; Abbrev [2] 0x8d:0xa DW_TAG_variable
	.long	151                             ; DW_AT_type
	.byte	1                               ; DW_AT_decl_file
	.byte	60                              ; DW_AT_decl_line
	.byte	2                               ; DW_AT_location
	.byte	161
	.byte	4
	.byte	3                               ; Abbrev [3] 0x97:0xc DW_TAG_array_type
	.long	67                              ; DW_AT_type
	.byte	4                               ; Abbrev [4] 0x9c:0x6 DW_TAG_subrange_type
	.long	71                              ; DW_AT_type
	.byte	41                              ; DW_AT_count
	.byte	0                               ; End Of Children Mark
	.byte	2                               ; Abbrev [2] 0xa3:0xa DW_TAG_variable
	.long	55                              ; DW_AT_type
	.byte	1                               ; DW_AT_decl_file
	.byte	65                              ; DW_AT_decl_line
	.byte	2                               ; DW_AT_location
	.byte	161
	.byte	5
	.byte	2                               ; Abbrev [2] 0xad:0xa DW_TAG_variable
	.long	183                             ; DW_AT_type
	.byte	1                               ; DW_AT_decl_file
	.byte	67                              ; DW_AT_decl_line
	.byte	2                               ; DW_AT_location
	.byte	161
	.byte	6
	.byte	3                               ; Abbrev [3] 0xb7:0xc DW_TAG_array_type
	.long	67                              ; DW_AT_type
	.byte	4                               ; Abbrev [4] 0xbc:0x6 DW_TAG_subrange_type
	.long	71                              ; DW_AT_type
	.byte	27                              ; DW_AT_count
	.byte	0                               ; End Of Children Mark
	.byte	2                               ; Abbrev [2] 0xc3:0xa DW_TAG_variable
	.long	205                             ; DW_AT_type
	.byte	1                               ; DW_AT_decl_file
	.byte	71                              ; DW_AT_decl_line
	.byte	2                               ; DW_AT_location
	.byte	161
	.byte	7
	.byte	3                               ; Abbrev [3] 0xcd:0xc DW_TAG_array_type
	.long	67                              ; DW_AT_type
	.byte	4                               ; Abbrev [4] 0xd2:0x6 DW_TAG_subrange_type
	.long	71                              ; DW_AT_type
	.byte	4                               ; DW_AT_count
	.byte	0                               ; End Of Children Mark
	.byte	2                               ; Abbrev [2] 0xd9:0xa DW_TAG_variable
	.long	227                             ; DW_AT_type
	.byte	1                               ; DW_AT_decl_file
	.byte	85                              ; DW_AT_decl_line
	.byte	2                               ; DW_AT_location
	.byte	161
	.byte	8
	.byte	3                               ; Abbrev [3] 0xe3:0xc DW_TAG_array_type
	.long	67                              ; DW_AT_type
	.byte	4                               ; Abbrev [4] 0xe8:0x6 DW_TAG_subrange_type
	.long	71                              ; DW_AT_type
	.byte	16                              ; DW_AT_count
	.byte	0                               ; End Of Children Mark
	.byte	2                               ; Abbrev [2] 0xef:0xa DW_TAG_variable
	.long	249                             ; DW_AT_type
	.byte	1                               ; DW_AT_decl_file
	.byte	90                              ; DW_AT_decl_line
	.byte	2                               ; DW_AT_location
	.byte	161
	.byte	9
	.byte	3                               ; Abbrev [3] 0xf9:0xc DW_TAG_array_type
	.long	67                              ; DW_AT_type
	.byte	4                               ; Abbrev [4] 0xfe:0x6 DW_TAG_subrange_type
	.long	71                              ; DW_AT_type
	.byte	26                              ; DW_AT_count
	.byte	0                               ; End Of Children Mark
	.byte	2                               ; Abbrev [2] 0x105:0xa DW_TAG_variable
	.long	271                             ; DW_AT_type
	.byte	1                               ; DW_AT_decl_file
	.byte	94                              ; DW_AT_decl_line
	.byte	2                               ; DW_AT_location
	.byte	161
	.byte	10
	.byte	3                               ; Abbrev [3] 0x10f:0xc DW_TAG_array_type
	.long	67                              ; DW_AT_type
	.byte	4                               ; Abbrev [4] 0x114:0x6 DW_TAG_subrange_type
	.long	71                              ; DW_AT_type
	.byte	15                              ; DW_AT_count
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x11b:0x7 DW_TAG_variable
	.long	290                             ; DW_AT_type
	.byte	1                               ; DW_AT_decl_file
	.byte	96                              ; DW_AT_decl_line
	.byte	3                               ; Abbrev [3] 0x122:0xc DW_TAG_array_type
	.long	67                              ; DW_AT_type
	.byte	4                               ; Abbrev [4] 0x127:0x6 DW_TAG_subrange_type
	.long	71                              ; DW_AT_type
	.byte	19                              ; DW_AT_count
	.byte	0                               ; End Of Children Mark
	.byte	2                               ; Abbrev [2] 0x12e:0xa DW_TAG_variable
	.long	205                             ; DW_AT_type
	.byte	1                               ; DW_AT_decl_file
	.byte	98                              ; DW_AT_decl_line
	.byte	2                               ; DW_AT_location
	.byte	161
	.byte	11
	.byte	7                               ; Abbrev [7] 0x138:0x7 DW_TAG_variable
	.long	55                              ; DW_AT_type
	.byte	1                               ; DW_AT_decl_file
	.byte	100                             ; DW_AT_decl_line
	.byte	2                               ; Abbrev [2] 0x13f:0xa DW_TAG_variable
	.long	329                             ; DW_AT_type
	.byte	1                               ; DW_AT_decl_file
	.byte	107                             ; DW_AT_decl_line
	.byte	2                               ; DW_AT_location
	.byte	161
	.byte	12
	.byte	3                               ; Abbrev [3] 0x149:0xc DW_TAG_array_type
	.long	67                              ; DW_AT_type
	.byte	4                               ; Abbrev [4] 0x14e:0x6 DW_TAG_subrange_type
	.long	71                              ; DW_AT_type
	.byte	18                              ; DW_AT_count
	.byte	0                               ; End Of Children Mark
	.byte	8                               ; Abbrev [8] 0x155:0x95 DW_TAG_subprogram
	.byte	13                              ; DW_AT_low_pc
.set Lset26, Lfunc_end0-Lfunc_begin0    ; DW_AT_high_pc
	.long	Lset26
	.byte	1                               ; DW_AT_frame_base
	.byte	109
                                        ; DW_AT_call_all_calls
	.byte	62                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	44                              ; DW_AT_decl_line
                                        ; DW_AT_prototyped
                                        ; DW_AT_external
                                        ; DW_AT_APPLE_optimized
	.byte	9                               ; Abbrev [9] 0x160:0x9 DW_TAG_formal_parameter
	.byte	0                               ; DW_AT_location
	.byte	48                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	44                              ; DW_AT_decl_line
	.long	878                             ; DW_AT_type
	.byte	9                               ; Abbrev [9] 0x169:0x9 DW_TAG_formal_parameter
	.byte	1                               ; DW_AT_location
	.byte	69                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	44                              ; DW_AT_decl_line
	.long	1170                            ; DW_AT_type
	.byte	9                               ; Abbrev [9] 0x172:0x9 DW_TAG_formal_parameter
	.byte	2                               ; DW_AT_location
	.byte	52                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	44                              ; DW_AT_decl_line
	.long	718                             ; DW_AT_type
	.byte	10                              ; Abbrev [10] 0x17b:0x9 DW_TAG_variable
	.byte	3                               ; DW_AT_location
	.byte	53                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	45                              ; DW_AT_decl_line
	.long	510                             ; DW_AT_type
	.byte	11                              ; Abbrev [11] 0x184:0xc DW_TAG_lexical_block
	.byte	0                               ; DW_AT_ranges
	.byte	10                              ; Abbrev [10] 0x186:0x9 DW_TAG_variable
	.byte	4                               ; DW_AT_location
	.byte	54                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	51                              ; DW_AT_decl_line
	.long	718                             ; DW_AT_type
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x190:0xe DW_TAG_call_site
	.long	490                             ; DW_AT_call_origin
	.byte	14                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x196:0x7 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	3                               ; DW_AT_call_value
	.byte	163
	.byte	1
	.byte	80
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x19e:0x13 DW_TAG_call_site
	.long	924                             ; DW_AT_call_origin
	.byte	15                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x1a4:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	81
	.byte	2                               ; DW_AT_call_value
	.byte	134
	.byte	0
	.byte	13                              ; Abbrev [13] 0x1aa:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	2                               ; DW_AT_call_value
	.byte	131
	.byte	0
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x1b1:0xd DW_TAG_call_site
	.long	949                             ; DW_AT_call_origin
	.byte	16                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x1b7:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	2                               ; DW_AT_call_value
	.byte	131
	.byte	0
	.byte	0                               ; End Of Children Mark
	.byte	14                              ; Abbrev [14] 0x1be:0x6 DW_TAG_call_site
	.long	963                             ; DW_AT_call_origin
	.byte	17                              ; DW_AT_call_return_pc
	.byte	14                              ; Abbrev [14] 0x1c4:0x6 DW_TAG_call_site
	.long	978                             ; DW_AT_call_origin
	.byte	18                              ; DW_AT_call_return_pc
	.byte	12                              ; Abbrev [12] 0x1ca:0xd DW_TAG_call_site
	.long	949                             ; DW_AT_call_origin
	.byte	19                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x1d0:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	2                               ; DW_AT_call_value
	.byte	131
	.byte	0
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x1d7:0xc DW_TAG_call_site
	.long	998                             ; DW_AT_call_origin
	.byte	20                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x1dd:0x5 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	1                               ; DW_AT_call_value
	.byte	49
	.byte	0                               ; End Of Children Mark
	.byte	14                              ; Abbrev [14] 0x1e3:0x6 DW_TAG_call_site
	.long	1008                            ; DW_AT_call_origin
	.byte	21                              ; DW_AT_call_return_pc
	.byte	0                               ; End Of Children Mark
	.byte	15                              ; Abbrev [15] 0x1ea:0x14 DW_TAG_subprogram
	.byte	7                               ; DW_AT_linkage_name
	.byte	8                               ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.byte	240                             ; DW_AT_decl_line
                                        ; DW_AT_prototyped
	.long	510                             ; DW_AT_type
                                        ; DW_AT_declaration
                                        ; DW_AT_external
                                        ; DW_AT_APPLE_optimized
	.byte	16                              ; Abbrev [16] 0x1f3:0x5 DW_TAG_formal_parameter
	.long	919                             ; DW_AT_type
	.byte	16                              ; Abbrev [16] 0x1f8:0x5 DW_TAG_formal_parameter
	.long	919                             ; DW_AT_type
	.byte	0                               ; End Of Children Mark
	.byte	17                              ; Abbrev [17] 0x1fe:0x5 DW_TAG_pointer_type
	.long	515                             ; DW_AT_type
	.byte	18                              ; Abbrev [18] 0x203:0x8 DW_TAG_typedef
	.long	523                             ; DW_AT_type
	.byte	41                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.byte	159                             ; DW_AT_decl_line
	.byte	19                              ; Abbrev [19] 0x20b:0xba DW_TAG_structure_type
	.byte	40                              ; DW_AT_name
	.byte	152                             ; DW_AT_byte_size
	.byte	2                               ; DW_AT_decl_file
	.byte	128                             ; DW_AT_decl_line
	.byte	20                              ; Abbrev [20] 0x210:0x9 DW_TAG_member
	.byte	9                               ; DW_AT_name
	.long	709                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	129                             ; DW_AT_decl_line
	.byte	0                               ; DW_AT_data_member_location
	.byte	20                              ; Abbrev [20] 0x219:0x9 DW_TAG_member
	.byte	11                              ; DW_AT_name
	.long	718                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	130                             ; DW_AT_decl_line
	.byte	8                               ; DW_AT_data_member_location
	.byte	20                              ; Abbrev [20] 0x222:0x9 DW_TAG_member
	.byte	13                              ; DW_AT_name
	.long	718                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	131                             ; DW_AT_decl_line
	.byte	12                              ; DW_AT_data_member_location
	.byte	20                              ; Abbrev [20] 0x22b:0x9 DW_TAG_member
	.byte	14                              ; DW_AT_name
	.long	722                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	132                             ; DW_AT_decl_line
	.byte	16                              ; DW_AT_data_member_location
	.byte	20                              ; Abbrev [20] 0x234:0x9 DW_TAG_member
	.byte	16                              ; DW_AT_name
	.long	722                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	133                             ; DW_AT_decl_line
	.byte	18                              ; DW_AT_data_member_location
	.byte	20                              ; Abbrev [20] 0x23d:0x9 DW_TAG_member
	.byte	17                              ; DW_AT_name
	.long	726                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	134                             ; DW_AT_decl_line
	.byte	24                              ; DW_AT_data_member_location
	.byte	20                              ; Abbrev [20] 0x246:0x9 DW_TAG_member
	.byte	21                              ; DW_AT_name
	.long	718                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	135                             ; DW_AT_decl_line
	.byte	40                              ; DW_AT_data_member_location
	.byte	20                              ; Abbrev [20] 0x24f:0x9 DW_TAG_member
	.byte	22                              ; DW_AT_name
	.long	750                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	138                             ; DW_AT_decl_line
	.byte	48                              ; DW_AT_data_member_location
	.byte	20                              ; Abbrev [20] 0x258:0x9 DW_TAG_member
	.byte	23                              ; DW_AT_name
	.long	751                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	139                             ; DW_AT_decl_line
	.byte	56                              ; DW_AT_data_member_location
	.byte	20                              ; Abbrev [20] 0x261:0x9 DW_TAG_member
	.byte	24                              ; DW_AT_name
	.long	767                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	140                             ; DW_AT_decl_line
	.byte	64                              ; DW_AT_data_member_location
	.byte	20                              ; Abbrev [20] 0x26a:0x9 DW_TAG_member
	.byte	25                              ; DW_AT_name
	.long	798                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	141                             ; DW_AT_decl_line
	.byte	72                              ; DW_AT_data_member_location
	.byte	20                              ; Abbrev [20] 0x273:0x9 DW_TAG_member
	.byte	30                              ; DW_AT_name
	.long	852                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	142                             ; DW_AT_decl_line
	.byte	80                              ; DW_AT_data_member_location
	.byte	20                              ; Abbrev [20] 0x27c:0x9 DW_TAG_member
	.byte	31                              ; DW_AT_name
	.long	726                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	145                             ; DW_AT_decl_line
	.byte	88                              ; DW_AT_data_member_location
	.byte	20                              ; Abbrev [20] 0x285:0x9 DW_TAG_member
	.byte	32                              ; DW_AT_name
	.long	888                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	146                             ; DW_AT_decl_line
	.byte	104                             ; DW_AT_data_member_location
	.byte	20                              ; Abbrev [20] 0x28e:0x9 DW_TAG_member
	.byte	34                              ; DW_AT_name
	.long	718                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	147                             ; DW_AT_decl_line
	.byte	112                             ; DW_AT_data_member_location
	.byte	20                              ; Abbrev [20] 0x297:0x9 DW_TAG_member
	.byte	35                              ; DW_AT_name
	.long	895                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	150                             ; DW_AT_decl_line
	.byte	116                             ; DW_AT_data_member_location
	.byte	20                              ; Abbrev [20] 0x2a0:0x9 DW_TAG_member
	.byte	36                              ; DW_AT_name
	.long	907                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	151                             ; DW_AT_decl_line
	.byte	119                             ; DW_AT_data_member_location
	.byte	20                              ; Abbrev [20] 0x2a9:0x9 DW_TAG_member
	.byte	37                              ; DW_AT_name
	.long	726                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	154                             ; DW_AT_decl_line
	.byte	120                             ; DW_AT_data_member_location
	.byte	20                              ; Abbrev [20] 0x2b2:0x9 DW_TAG_member
	.byte	38                              ; DW_AT_name
	.long	718                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	157                             ; DW_AT_decl_line
	.byte	136                             ; DW_AT_data_member_location
	.byte	20                              ; Abbrev [20] 0x2bb:0x9 DW_TAG_member
	.byte	39                              ; DW_AT_name
	.long	824                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	158                             ; DW_AT_decl_line
	.byte	144                             ; DW_AT_data_member_location
	.byte	0                               ; End Of Children Mark
	.byte	17                              ; Abbrev [17] 0x2c5:0x5 DW_TAG_pointer_type
	.long	714                             ; DW_AT_type
	.byte	5                               ; Abbrev [5] 0x2ca:0x4 DW_TAG_base_type
	.byte	10                              ; DW_AT_name
	.byte	8                               ; DW_AT_encoding
	.byte	1                               ; DW_AT_byte_size
	.byte	5                               ; Abbrev [5] 0x2ce:0x4 DW_TAG_base_type
	.byte	12                              ; DW_AT_name
	.byte	5                               ; DW_AT_encoding
	.byte	4                               ; DW_AT_byte_size
	.byte	5                               ; Abbrev [5] 0x2d2:0x4 DW_TAG_base_type
	.byte	15                              ; DW_AT_name
	.byte	5                               ; DW_AT_encoding
	.byte	2                               ; DW_AT_byte_size
	.byte	19                              ; Abbrev [19] 0x2d6:0x18 DW_TAG_structure_type
	.byte	20                              ; DW_AT_name
	.byte	16                              ; DW_AT_byte_size
	.byte	2                               ; DW_AT_decl_file
	.byte	94                              ; DW_AT_decl_line
	.byte	20                              ; Abbrev [20] 0x2db:0x9 DW_TAG_member
	.byte	18                              ; DW_AT_name
	.long	709                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	95                              ; DW_AT_decl_line
	.byte	0                               ; DW_AT_data_member_location
	.byte	20                              ; Abbrev [20] 0x2e4:0x9 DW_TAG_member
	.byte	19                              ; DW_AT_name
	.long	718                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.byte	96                              ; DW_AT_decl_line
	.byte	8                               ; DW_AT_data_member_location
	.byte	0                               ; End Of Children Mark
	.byte	21                              ; Abbrev [21] 0x2ee:0x1 DW_TAG_pointer_type
	.byte	17                              ; Abbrev [17] 0x2ef:0x5 DW_TAG_pointer_type
	.long	756                             ; DW_AT_type
	.byte	22                              ; Abbrev [22] 0x2f4:0xb DW_TAG_subroutine_type
	.long	718                             ; DW_AT_type
                                        ; DW_AT_prototyped
	.byte	16                              ; Abbrev [16] 0x2f9:0x5 DW_TAG_formal_parameter
	.long	750                             ; DW_AT_type
	.byte	0                               ; End Of Children Mark
	.byte	17                              ; Abbrev [17] 0x2ff:0x5 DW_TAG_pointer_type
	.long	772                             ; DW_AT_type
	.byte	22                              ; Abbrev [22] 0x304:0x15 DW_TAG_subroutine_type
	.long	718                             ; DW_AT_type
                                        ; DW_AT_prototyped
	.byte	16                              ; Abbrev [16] 0x309:0x5 DW_TAG_formal_parameter
	.long	750                             ; DW_AT_type
	.byte	16                              ; Abbrev [16] 0x30e:0x5 DW_TAG_formal_parameter
	.long	793                             ; DW_AT_type
	.byte	16                              ; Abbrev [16] 0x313:0x5 DW_TAG_formal_parameter
	.long	718                             ; DW_AT_type
	.byte	0                               ; End Of Children Mark
	.byte	17                              ; Abbrev [17] 0x319:0x5 DW_TAG_pointer_type
	.long	67                              ; DW_AT_type
	.byte	17                              ; Abbrev [17] 0x31e:0x5 DW_TAG_pointer_type
	.long	803                             ; DW_AT_type
	.byte	22                              ; Abbrev [22] 0x323:0x15 DW_TAG_subroutine_type
	.long	824                             ; DW_AT_type
                                        ; DW_AT_prototyped
	.byte	16                              ; Abbrev [16] 0x328:0x5 DW_TAG_formal_parameter
	.long	750                             ; DW_AT_type
	.byte	16                              ; Abbrev [16] 0x32d:0x5 DW_TAG_formal_parameter
	.long	824                             ; DW_AT_type
	.byte	16                              ; Abbrev [16] 0x332:0x5 DW_TAG_formal_parameter
	.long	718                             ; DW_AT_type
	.byte	0                               ; End Of Children Mark
	.byte	18                              ; Abbrev [18] 0x338:0x8 DW_TAG_typedef
	.long	832                             ; DW_AT_type
	.byte	29                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.byte	83                              ; DW_AT_decl_line
	.byte	18                              ; Abbrev [18] 0x340:0x8 DW_TAG_typedef
	.long	840                             ; DW_AT_type
	.byte	28                              ; DW_AT_name
	.byte	4                               ; DW_AT_decl_file
	.byte	83                              ; DW_AT_decl_line
	.byte	18                              ; Abbrev [18] 0x348:0x8 DW_TAG_typedef
	.long	848                             ; DW_AT_type
	.byte	27                              ; DW_AT_name
	.byte	3                               ; DW_AT_decl_file
	.byte	37                              ; DW_AT_decl_line
	.byte	5                               ; Abbrev [5] 0x350:0x4 DW_TAG_base_type
	.byte	26                              ; DW_AT_name
	.byte	5                               ; DW_AT_encoding
	.byte	8                               ; DW_AT_byte_size
	.byte	17                              ; Abbrev [17] 0x354:0x5 DW_TAG_pointer_type
	.long	857                             ; DW_AT_type
	.byte	22                              ; Abbrev [22] 0x359:0x15 DW_TAG_subroutine_type
	.long	718                             ; DW_AT_type
                                        ; DW_AT_prototyped
	.byte	16                              ; Abbrev [16] 0x35e:0x5 DW_TAG_formal_parameter
	.long	750                             ; DW_AT_type
	.byte	16                              ; Abbrev [16] 0x363:0x5 DW_TAG_formal_parameter
	.long	878                             ; DW_AT_type
	.byte	16                              ; Abbrev [16] 0x368:0x5 DW_TAG_formal_parameter
	.long	718                             ; DW_AT_type
	.byte	0                               ; End Of Children Mark
	.byte	17                              ; Abbrev [17] 0x36e:0x5 DW_TAG_pointer_type
	.long	883                             ; DW_AT_type
	.byte	23                              ; Abbrev [23] 0x373:0x5 DW_TAG_const_type
	.long	67                              ; DW_AT_type
	.byte	17                              ; Abbrev [17] 0x378:0x5 DW_TAG_pointer_type
	.long	893                             ; DW_AT_type
	.byte	24                              ; Abbrev [24] 0x37d:0x2 DW_TAG_structure_type
	.byte	33                              ; DW_AT_name
                                        ; DW_AT_declaration
	.byte	3                               ; Abbrev [3] 0x37f:0xc DW_TAG_array_type
	.long	714                             ; DW_AT_type
	.byte	4                               ; Abbrev [4] 0x384:0x6 DW_TAG_subrange_type
	.long	71                              ; DW_AT_type
	.byte	3                               ; DW_AT_count
	.byte	0                               ; End Of Children Mark
	.byte	3                               ; Abbrev [3] 0x38b:0xc DW_TAG_array_type
	.long	714                             ; DW_AT_type
	.byte	4                               ; Abbrev [4] 0x390:0x6 DW_TAG_subrange_type
	.long	71                              ; DW_AT_type
	.byte	1                               ; DW_AT_count
	.byte	0                               ; End Of Children Mark
	.byte	25                              ; Abbrev [25] 0x397:0x5 DW_TAG_restrict_type
	.long	878                             ; DW_AT_type
	.byte	26                              ; Abbrev [26] 0x39c:0x14 DW_TAG_subprogram
	.byte	42                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.byte	248                             ; DW_AT_decl_line
                                        ; DW_AT_prototyped
	.long	718                             ; DW_AT_type
                                        ; DW_AT_declaration
                                        ; DW_AT_external
                                        ; DW_AT_APPLE_optimized
	.byte	16                              ; Abbrev [16] 0x3a4:0x5 DW_TAG_formal_parameter
	.long	944                             ; DW_AT_type
	.byte	16                              ; Abbrev [16] 0x3a9:0x5 DW_TAG_formal_parameter
	.long	919                             ; DW_AT_type
	.byte	27                              ; Abbrev [27] 0x3ae:0x1 DW_TAG_unspecified_parameters
	.byte	0                               ; End Of Children Mark
	.byte	25                              ; Abbrev [25] 0x3b0:0x5 DW_TAG_restrict_type
	.long	510                             ; DW_AT_type
	.byte	26                              ; Abbrev [26] 0x3b5:0xe DW_TAG_subprogram
	.byte	43                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.byte	230                             ; DW_AT_decl_line
                                        ; DW_AT_prototyped
	.long	718                             ; DW_AT_type
                                        ; DW_AT_declaration
                                        ; DW_AT_external
                                        ; DW_AT_APPLE_optimized
	.byte	16                              ; Abbrev [16] 0x3bd:0x5 DW_TAG_formal_parameter
	.long	510                             ; DW_AT_type
	.byte	0                               ; End Of Children Mark
	.byte	26                              ; Abbrev [26] 0x3c3:0xf DW_TAG_subprogram
	.byte	44                              ; DW_AT_name
	.byte	5                               ; DW_AT_decl_file
	.byte	31                              ; DW_AT_decl_line
                                        ; DW_AT_prototyped
	.long	718                             ; DW_AT_type
                                        ; DW_AT_declaration
                                        ; DW_AT_external
                                        ; DW_AT_APPLE_optimized
	.byte	16                              ; Abbrev [16] 0x3cb:0x5 DW_TAG_formal_parameter
	.long	919                             ; DW_AT_type
	.byte	27                              ; Abbrev [27] 0x3d0:0x1 DW_TAG_unspecified_parameters
	.byte	0                               ; End Of Children Mark
	.byte	26                              ; Abbrev [26] 0x3d2:0x14 DW_TAG_subprogram
	.byte	45                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.byte	242                             ; DW_AT_decl_line
                                        ; DW_AT_prototyped
	.long	718                             ; DW_AT_type
                                        ; DW_AT_declaration
                                        ; DW_AT_external
                                        ; DW_AT_APPLE_optimized
	.byte	16                              ; Abbrev [16] 0x3da:0x5 DW_TAG_formal_parameter
	.long	944                             ; DW_AT_type
	.byte	16                              ; Abbrev [16] 0x3df:0x5 DW_TAG_formal_parameter
	.long	919                             ; DW_AT_type
	.byte	27                              ; Abbrev [27] 0x3e4:0x1 DW_TAG_unspecified_parameters
	.byte	0                               ; End Of Children Mark
	.byte	28                              ; Abbrev [28] 0x3e6:0xa DW_TAG_subprogram
	.byte	46                              ; DW_AT_name
	.byte	6                               ; DW_AT_decl_file
	.byte	141                             ; DW_AT_decl_line
                                        ; DW_AT_prototyped
                                        ; DW_AT_declaration
                                        ; DW_AT_external
                                        ; DW_AT_APPLE_optimized
                                        ; DW_AT_noreturn
	.byte	16                              ; Abbrev [16] 0x3ea:0x5 DW_TAG_formal_parameter
	.long	718                             ; DW_AT_type
	.byte	0                               ; End Of Children Mark
	.byte	29                              ; Abbrev [29] 0x3f0:0xa DW_TAG_subprogram
	.byte	53                              ; DW_AT_low_pc
.set Lset27, Lfunc_end3-Lfunc_begin3    ; DW_AT_high_pc
	.long	Lset27
	.byte	1                               ; DW_AT_frame_base
	.byte	109
	.byte	64                              ; DW_AT_linkage_name
	.byte	64                              ; DW_AT_name
                                        ; DW_AT_APPLE_optimized
	.byte	30                              ; Abbrev [30] 0x3fa:0x5f DW_TAG_subprogram
	.byte	22                              ; DW_AT_low_pc
.set Lset28, Lfunc_end1-Lfunc_begin1    ; DW_AT_high_pc
	.long	Lset28
	.byte	1                               ; DW_AT_frame_base
	.byte	109
                                        ; DW_AT_call_all_calls
	.long	1123                            ; DW_AT_abstract_origin
	.byte	31                              ; Abbrev [31] 0x406:0x6 DW_TAG_formal_parameter
	.byte	5                               ; DW_AT_location
	.long	1127                            ; DW_AT_abstract_origin
	.byte	31                              ; Abbrev [31] 0x40c:0x6 DW_TAG_formal_parameter
	.byte	6                               ; DW_AT_location
	.long	1135                            ; DW_AT_abstract_origin
	.byte	31                              ; Abbrev [31] 0x412:0x6 DW_TAG_formal_parameter
	.byte	7                               ; DW_AT_location
	.long	1143                            ; DW_AT_abstract_origin
	.byte	32                              ; Abbrev [32] 0x418:0x6 DW_TAG_variable
	.byte	8                               ; DW_AT_location
	.long	1151                            ; DW_AT_abstract_origin
	.byte	33                              ; Abbrev [33] 0x41e:0xd DW_TAG_lexical_block
	.byte	23                              ; DW_AT_low_pc
.set Lset29, Ltmp34-Ltmp27              ; DW_AT_high_pc
	.long	Lset29
	.byte	32                              ; Abbrev [32] 0x424:0x6 DW_TAG_variable
	.byte	9                               ; DW_AT_location
	.long	1160                            ; DW_AT_abstract_origin
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x42b:0xe DW_TAG_call_site
	.long	490                             ; DW_AT_call_origin
	.byte	24                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x431:0x7 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	3                               ; DW_AT_call_value
	.byte	163
	.byte	1
	.byte	80
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x439:0x13 DW_TAG_call_site
	.long	978                             ; DW_AT_call_origin
	.byte	25                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x43f:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	81
	.byte	2                               ; DW_AT_call_value
	.byte	133
	.byte	0
	.byte	13                              ; Abbrev [13] 0x445:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	2                               ; DW_AT_call_value
	.byte	132
	.byte	0
	.byte	0                               ; End Of Children Mark
	.byte	34                              ; Abbrev [34] 0x44c:0x6 DW_TAG_call_site
	.long	949                             ; DW_AT_call_origin
                                        ; DW_AT_call_tail_call
	.byte	26                              ; DW_AT_call_pc
	.byte	14                              ; Abbrev [14] 0x452:0x6 DW_TAG_call_site
	.long	1113                            ; DW_AT_call_origin
	.byte	27                              ; DW_AT_call_return_pc
	.byte	0                               ; End Of Children Mark
	.byte	29                              ; Abbrev [29] 0x459:0xa DW_TAG_subprogram
	.byte	54                              ; DW_AT_low_pc
.set Lset30, Lfunc_end4-Lfunc_begin4    ; DW_AT_high_pc
	.long	Lset30
	.byte	1                               ; DW_AT_frame_base
	.byte	109
	.byte	65                              ; DW_AT_linkage_name
	.byte	65                              ; DW_AT_name
                                        ; DW_AT_APPLE_optimized
	.byte	35                              ; Abbrev [35] 0x463:0x2f DW_TAG_subprogram
	.byte	47                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	64                              ; DW_AT_decl_line
                                        ; DW_AT_prototyped
                                        ; DW_AT_external
                                        ; DW_AT_APPLE_optimized
                                        ; DW_AT_inline
	.byte	36                              ; Abbrev [36] 0x467:0x8 DW_TAG_formal_parameter
	.byte	48                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	64                              ; DW_AT_decl_line
	.long	878                             ; DW_AT_type
	.byte	36                              ; Abbrev [36] 0x46f:0x8 DW_TAG_formal_parameter
	.byte	49                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	64                              ; DW_AT_decl_line
	.long	1170                            ; DW_AT_type
	.byte	36                              ; Abbrev [36] 0x477:0x8 DW_TAG_formal_parameter
	.byte	52                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	64                              ; DW_AT_decl_line
	.long	718                             ; DW_AT_type
	.byte	37                              ; Abbrev [37] 0x47f:0x8 DW_TAG_variable
	.byte	53                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	65                              ; DW_AT_decl_line
	.long	510                             ; DW_AT_type
	.byte	38                              ; Abbrev [38] 0x487:0xa DW_TAG_lexical_block
	.byte	37                              ; Abbrev [37] 0x488:0x8 DW_TAG_variable
	.byte	54                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	70                              ; DW_AT_decl_line
	.long	718                             ; DW_AT_type
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	17                              ; Abbrev [17] 0x492:0x5 DW_TAG_pointer_type
	.long	1175                            ; DW_AT_type
	.byte	18                              ; Abbrev [18] 0x497:0x8 DW_TAG_typedef
	.long	1183                            ; DW_AT_type
	.byte	51                              ; DW_AT_name
	.byte	7                               ; DW_AT_decl_file
	.byte	30                              ; DW_AT_decl_line
	.byte	5                               ; Abbrev [5] 0x49f:0x4 DW_TAG_base_type
	.byte	50                              ; DW_AT_name
	.byte	6                               ; DW_AT_encoding
	.byte	1                               ; DW_AT_byte_size
	.byte	39                              ; Abbrev [39] 0x4a3:0x197 DW_TAG_subprogram
	.byte	28                              ; DW_AT_low_pc
.set Lset31, Lfunc_end2-Lfunc_begin2    ; DW_AT_high_pc
	.long	Lset31
	.byte	1                               ; DW_AT_frame_base
	.byte	109
                                        ; DW_AT_call_all_calls
	.byte	63                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	77                              ; DW_AT_decl_line
	.long	718                             ; DW_AT_type
                                        ; DW_AT_external
                                        ; DW_AT_APPLE_optimized
	.byte	40                              ; Abbrev [40] 0x4b2:0xc DW_TAG_variable
	.ascii	"\200\200\200\002"              ; DW_AT_const_value
	.byte	70                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	78                              ; DW_AT_decl_line
	.long	718                             ; DW_AT_type
	.byte	40                              ; Abbrev [40] 0x4be:0xc DW_TAG_variable
	.ascii	"\200\200\200\004"              ; DW_AT_const_value
	.byte	71                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	79                              ; DW_AT_decl_line
	.long	718                             ; DW_AT_type
	.byte	10                              ; Abbrev [10] 0x4ca:0x9 DW_TAG_variable
	.byte	10                              ; DW_AT_location
	.byte	69                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	81                              ; DW_AT_decl_line
	.long	1170                            ; DW_AT_type
	.byte	10                              ; Abbrev [10] 0x4d3:0x9 DW_TAG_variable
	.byte	11                              ; DW_AT_location
	.byte	49                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	82                              ; DW_AT_decl_line
	.long	1170                            ; DW_AT_type
	.byte	33                              ; Abbrev [33] 0x4dc:0x10 DW_TAG_lexical_block
	.byte	29                              ; DW_AT_low_pc
.set Lset32, Ltmp59-Ltmp49              ; DW_AT_high_pc
	.long	Lset32
	.byte	10                              ; Abbrev [10] 0x4e2:0x9 DW_TAG_variable
	.byte	12                              ; DW_AT_location
	.byte	54                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	97                              ; DW_AT_decl_line
	.long	718                             ; DW_AT_type
	.byte	0                               ; End Of Children Mark
	.byte	41                              ; Abbrev [41] 0x4ec:0x30 DW_TAG_inlined_subroutine
	.long	1123                            ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	107                             ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	31                              ; Abbrev [31] 0x4f5:0x6 DW_TAG_formal_parameter
	.byte	15                              ; DW_AT_location
	.long	1127                            ; DW_AT_abstract_origin
	.byte	31                              ; Abbrev [31] 0x4fb:0x6 DW_TAG_formal_parameter
	.byte	14                              ; DW_AT_location
	.long	1135                            ; DW_AT_abstract_origin
	.byte	31                              ; Abbrev [31] 0x501:0x6 DW_TAG_formal_parameter
	.byte	13                              ; DW_AT_location
	.long	1143                            ; DW_AT_abstract_origin
	.byte	32                              ; Abbrev [32] 0x507:0x6 DW_TAG_variable
	.byte	16                              ; DW_AT_location
	.long	1151                            ; DW_AT_abstract_origin
	.byte	33                              ; Abbrev [33] 0x50d:0xe DW_TAG_lexical_block
	.byte	30                              ; DW_AT_low_pc
.set Lset33, Ltmp71-Ltmp68              ; DW_AT_high_pc
	.long	Lset33
	.byte	42                              ; Abbrev [42] 0x513:0x7 DW_TAG_variable
	.byte	1                               ; DW_AT_location
	.byte	103
	.long	1160                            ; DW_AT_abstract_origin
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x51c:0x10 DW_TAG_call_site
	.long	1594                            ; DW_AT_call_origin
	.byte	31                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x522:0x9 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	5                               ; DW_AT_call_value
	.byte	16
	.ascii	"\200\200\200\002"
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x52c:0x10 DW_TAG_call_site
	.long	1594                            ; DW_AT_call_origin
	.byte	32                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x532:0x9 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	5                               ; DW_AT_call_value
	.byte	16
	.ascii	"\200\200\200\004"
	.byte	0                               ; End Of Children Mark
	.byte	14                              ; Abbrev [14] 0x53c:0x6 DW_TAG_call_site
	.long	963                             ; DW_AT_call_origin
	.byte	33                              ; DW_AT_call_return_pc
	.byte	12                              ; Abbrev [12] 0x542:0x16 DW_TAG_call_site
	.long	341                             ; DW_AT_call_origin
	.byte	34                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x548:0x9 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	82
	.byte	5                               ; DW_AT_call_value
	.byte	16
	.ascii	"\200\200\200\002"
	.byte	13                              ; Abbrev [13] 0x551:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	81
	.byte	2                               ; DW_AT_call_value
	.byte	131
	.byte	0
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x558:0xd DW_TAG_call_site
	.long	963                             ; DW_AT_call_origin
	.byte	35                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x55e:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	2                               ; DW_AT_call_value
	.byte	133
	.byte	0
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x565:0xd DW_TAG_call_site
	.long	963                             ; DW_AT_call_origin
	.byte	36                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x56b:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	2                               ; DW_AT_call_value
	.byte	133
	.byte	0
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x572:0xd DW_TAG_call_site
	.long	963                             ; DW_AT_call_origin
	.byte	37                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x578:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	2                               ; DW_AT_call_value
	.byte	133
	.byte	0
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x57f:0xd DW_TAG_call_site
	.long	963                             ; DW_AT_call_origin
	.byte	38                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x585:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	2                               ; DW_AT_call_value
	.byte	133
	.byte	0
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x58c:0xd DW_TAG_call_site
	.long	963                             ; DW_AT_call_origin
	.byte	39                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x592:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	2                               ; DW_AT_call_value
	.byte	133
	.byte	0
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x599:0xd DW_TAG_call_site
	.long	963                             ; DW_AT_call_origin
	.byte	40                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x59f:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	2                               ; DW_AT_call_value
	.byte	133
	.byte	0
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x5a6:0xd DW_TAG_call_site
	.long	963                             ; DW_AT_call_origin
	.byte	41                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x5ac:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	2                               ; DW_AT_call_value
	.byte	133
	.byte	0
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x5b3:0xd DW_TAG_call_site
	.long	963                             ; DW_AT_call_origin
	.byte	42                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x5b9:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	2                               ; DW_AT_call_value
	.byte	133
	.byte	0
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x5c0:0xd DW_TAG_call_site
	.long	963                             ; DW_AT_call_origin
	.byte	43                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x5c6:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	2                               ; DW_AT_call_value
	.byte	133
	.byte	0
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x5cd:0xd DW_TAG_call_site
	.long	963                             ; DW_AT_call_origin
	.byte	44                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x5d3:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	2                               ; DW_AT_call_value
	.byte	133
	.byte	0
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x5da:0x13 DW_TAG_call_site
	.long	1628                            ; DW_AT_call_origin
	.byte	45                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x5e0:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	81
	.byte	2                               ; DW_AT_call_value
	.byte	132
	.byte	0
	.byte	13                              ; Abbrev [13] 0x5e6:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	2                               ; DW_AT_call_value
	.byte	131
	.byte	0
	.byte	0                               ; End Of Children Mark
	.byte	14                              ; Abbrev [14] 0x5ed:0x6 DW_TAG_call_site
	.long	490                             ; DW_AT_call_origin
	.byte	46                              ; DW_AT_call_return_pc
	.byte	12                              ; Abbrev [12] 0x5f3:0x13 DW_TAG_call_site
	.long	978                             ; DW_AT_call_origin
	.byte	47                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x5f9:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	81
	.byte	2                               ; DW_AT_call_value
	.byte	134
	.byte	0
	.byte	13                              ; Abbrev [13] 0x5ff:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	2                               ; DW_AT_call_value
	.byte	133
	.byte	0
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x606:0xd DW_TAG_call_site
	.long	949                             ; DW_AT_call_origin
	.byte	48                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x60c:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	2                               ; DW_AT_call_value
	.byte	133
	.byte	0
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x613:0xd DW_TAG_call_site
	.long	1643                            ; DW_AT_call_origin
	.byte	49                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x619:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	2                               ; DW_AT_call_value
	.byte	131
	.byte	0
	.byte	0                               ; End Of Children Mark
	.byte	12                              ; Abbrev [12] 0x620:0xd DW_TAG_call_site
	.long	1643                            ; DW_AT_call_origin
	.byte	50                              ; DW_AT_call_return_pc
	.byte	13                              ; Abbrev [13] 0x626:0x6 DW_TAG_call_site_parameter
	.byte	1                               ; DW_AT_location
	.byte	80
	.byte	2                               ; DW_AT_call_value
	.byte	132
	.byte	0
	.byte	0                               ; End Of Children Mark
	.byte	14                              ; Abbrev [14] 0x62d:0x6 DW_TAG_call_site
	.long	1653                            ; DW_AT_call_origin
	.byte	51                              ; DW_AT_call_return_pc
	.byte	14                              ; Abbrev [14] 0x633:0x6 DW_TAG_call_site
	.long	1664                            ; DW_AT_call_origin
	.byte	52                              ; DW_AT_call_return_pc
	.byte	0                               ; End Of Children Mark
	.byte	26                              ; Abbrev [26] 0x63a:0xe DW_TAG_subprogram
	.byte	55                              ; DW_AT_name
	.byte	8                               ; DW_AT_decl_file
	.byte	54                              ; DW_AT_decl_line
                                        ; DW_AT_prototyped
	.long	750                             ; DW_AT_type
                                        ; DW_AT_declaration
                                        ; DW_AT_external
                                        ; DW_AT_APPLE_optimized
	.byte	16                              ; Abbrev [16] 0x642:0x5 DW_TAG_formal_parameter
	.long	1608                            ; DW_AT_type
	.byte	0                               ; End Of Children Mark
	.byte	18                              ; Abbrev [18] 0x648:0x8 DW_TAG_typedef
	.long	1616                            ; DW_AT_type
	.byte	58                              ; DW_AT_name
	.byte	9                               ; DW_AT_decl_file
	.byte	50                              ; DW_AT_decl_line
	.byte	18                              ; Abbrev [18] 0x650:0x8 DW_TAG_typedef
	.long	1624                            ; DW_AT_type
	.byte	57                              ; DW_AT_name
	.byte	3                               ; DW_AT_decl_file
	.byte	87                              ; DW_AT_decl_line
	.byte	5                               ; Abbrev [5] 0x658:0x4 DW_TAG_base_type
	.byte	56                              ; DW_AT_name
	.byte	7                               ; DW_AT_encoding
	.byte	8                               ; DW_AT_byte_size
	.byte	43                              ; Abbrev [43] 0x65c:0xf DW_TAG_subprogram
	.byte	59                              ; DW_AT_name
	.byte	10                              ; DW_AT_decl_file
	.byte	10                              ; DW_AT_decl_line
                                        ; DW_AT_prototyped
                                        ; DW_AT_declaration
                                        ; DW_AT_external
                                        ; DW_AT_APPLE_optimized
	.byte	16                              ; Abbrev [16] 0x660:0x5 DW_TAG_formal_parameter
	.long	1170                            ; DW_AT_type
	.byte	16                              ; Abbrev [16] 0x665:0x5 DW_TAG_formal_parameter
	.long	1170                            ; DW_AT_type
	.byte	0                               ; End Of Children Mark
	.byte	43                              ; Abbrev [43] 0x66b:0xa DW_TAG_subprogram
	.byte	60                              ; DW_AT_name
	.byte	8                               ; DW_AT_decl_file
	.byte	56                              ; DW_AT_decl_line
                                        ; DW_AT_prototyped
                                        ; DW_AT_declaration
                                        ; DW_AT_external
                                        ; DW_AT_APPLE_optimized
	.byte	16                              ; Abbrev [16] 0x66f:0x5 DW_TAG_formal_parameter
	.long	750                             ; DW_AT_type
	.byte	0                               ; End Of Children Mark
	.byte	44                              ; Abbrev [44] 0x675:0xb DW_TAG_subprogram
	.byte	61                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.short	261                             ; DW_AT_decl_line
                                        ; DW_AT_prototyped
                                        ; DW_AT_declaration
                                        ; DW_AT_external
                                        ; DW_AT_APPLE_optimized
	.byte	16                              ; Abbrev [16] 0x67a:0x5 DW_TAG_formal_parameter
	.long	878                             ; DW_AT_type
	.byte	0                               ; End Of Children Mark
	.byte	45                              ; Abbrev [45] 0x680:0x18 DW_TAG_subprogram
	.byte	55                              ; DW_AT_low_pc
.set Lset34, Lfunc_end5-Lfunc_begin5    ; DW_AT_high_pc
	.long	Lset34
	.byte	1                               ; DW_AT_frame_base
	.byte	109
	.byte	66                              ; DW_AT_linkage_name
	.byte	66                              ; DW_AT_name
                                        ; DW_AT_APPLE_optimized
	.byte	46                              ; Abbrev [46] 0x68a:0xd DW_TAG_inlined_subroutine
	.long	1123                            ; DW_AT_abstract_origin
	.byte	56                              ; DW_AT_low_pc
.set Lset35, Ltmp88-Ltmp87              ; DW_AT_high_pc
	.long	Lset35
	.byte	1                               ; DW_AT_call_file
	.byte	107                             ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	47                              ; Abbrev [47] 0x698:0xa DW_TAG_subprogram
	.byte	57                              ; DW_AT_low_pc
.set Lset36, Lfunc_end6-Lfunc_begin6    ; DW_AT_high_pc
	.long	Lset36
                                        ; DW_AT_APPLE_omit_frame_ptr
	.byte	1                               ; DW_AT_frame_base
	.byte	109
	.byte	67                              ; DW_AT_linkage_name
	.byte	68                              ; DW_AT_name
                                        ; DW_AT_artificial
                                        ; DW_AT_external
                                        ; DW_AT_APPLE_optimized
	.byte	0                               ; End Of Children Mark
Ldebug_info_end0:
	.section	__DWARF,__debug_rnglists,regular,debug
Ldebug_range0:
.set Lset37, Ldebug_list_header_end1-Ldebug_list_header_start1 ; Length
	.long	Lset37
Ldebug_list_header_start1:
	.short	5                               ; Version
	.byte	8                               ; Address size
	.byte	0                               ; Segment selector size
	.long	2                               ; Offset entry count
Lrnglists_table_base0:
.set Lset38, Ldebug_ranges0-Lrnglists_table_base0
	.long	Lset38
.set Lset39, Ldebug_ranges1-Lrnglists_table_base0
	.long	Lset39
Ldebug_ranges0:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 Ltmp5-Lfunc_begin0             ;   starting offset
	.uleb128 Ltmp13-Lfunc_begin0            ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 Ltmp18-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp21-Lfunc_begin0            ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
Ldebug_ranges1:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 Ltmp63-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp72-Lfunc_begin0            ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 Ltmp81-Lfunc_begin0            ;   starting offset
	.uleb128 Ltmp82-Lfunc_begin0            ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
Ldebug_list_header_end1:
	.section	__DWARF,__debug_str_offs,regular,debug
Lsection_str_off:
	.long	292                             ; Length of String Offsets Set
	.short	5
	.short	0
Lstr_offsets_base0:
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.asciz	"Apple clang version 16.0.0 (clang-1600.0.26.6)" ; string offset=0
	.asciz	"/Users/lmwgsrwny/Documents/_E8/C_Lang_Mamba/src/main.c" ; string offset=47
	.asciz	"/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk" ; string offset=102
	.asciz	"MacOSX15.2.sdk"                ; string offset=158
	.asciz	"/Users/lmwgsrwny/Documents/_E8/C_Lang_Mamba/src/build" ; string offset=173
	.asciz	"char"                          ; string offset=227
	.asciz	"__ARRAY_SIZE_TYPE__"           ; string offset=232
	.asciz	"read_input_from_file"          ; string offset=252
	.asciz	"_fopen"                        ; string offset=273
	.asciz	"fopen"                         ; string offset=280
	.asciz	"FILE"                          ; string offset=286
	.asciz	"__sFILE"                       ; string offset=291
	.asciz	"_p"                            ; string offset=299
	.asciz	"unsigned char"                 ; string offset=302
	.asciz	"_r"                            ; string offset=316
	.asciz	"int"                           ; string offset=319
	.asciz	"_w"                            ; string offset=323
	.asciz	"_flags"                        ; string offset=326
	.asciz	"short"                         ; string offset=333
	.asciz	"_file"                         ; string offset=339
	.asciz	"_bf"                           ; string offset=345
	.asciz	"__sbuf"                        ; string offset=349
	.asciz	"_base"                         ; string offset=356
	.asciz	"_size"                         ; string offset=362
	.asciz	"_lbfsize"                      ; string offset=368
	.asciz	"_cookie"                       ; string offset=377
	.asciz	"_close"                        ; string offset=385
	.asciz	"_read"                         ; string offset=392
	.asciz	"_seek"                         ; string offset=398
	.asciz	"fpos_t"                        ; string offset=404
	.asciz	"__darwin_off_t"                ; string offset=411
	.asciz	"__int64_t"                     ; string offset=426
	.asciz	"long long"                     ; string offset=436
	.asciz	"_write"                        ; string offset=446
	.asciz	"_ub"                           ; string offset=453
	.asciz	"_extra"                        ; string offset=457
	.asciz	"__sFILEX"                      ; string offset=464
	.asciz	"_ur"                           ; string offset=473
	.asciz	"_ubuf"                         ; string offset=477
	.asciz	"_nbuf"                         ; string offset=483
	.asciz	"_lb"                           ; string offset=489
	.asciz	"_blksize"                      ; string offset=493
	.asciz	"_offset"                       ; string offset=502
	.asciz	"fscanf"                        ; string offset=510
	.asciz	"fclose"                        ; string offset=517
	.asciz	"printf"                        ; string offset=524
	.asciz	"fprintf"                       ; string offset=531
	.asciz	"exit"                          ; string offset=539
	.asciz	"write_output_to_file"          ; string offset=544
	.asciz	"filename"                      ; string offset=565
	.asciz	"output"                        ; string offset=574
	.asciz	"int8_t"                        ; string offset=581
	.asciz	"signed char"                   ; string offset=588
	.asciz	"size"                          ; string offset=600
	.asciz	"file"                          ; string offset=605
	.asciz	"i"                             ; string offset=610
	.asciz	"main"                          ; string offset=612
	.asciz	"malloc"                        ; string offset=617
	.asciz	"size_t"                        ; string offset=624
	.asciz	"__darwin_size_t"               ; string offset=631
	.asciz	"unsigned long"                 ; string offset=647
	.asciz	"forward"                       ; string offset=661
	.asciz	"free"                          ; string offset=669
	.asciz	"perror"                        ; string offset=674
	.asciz	"read_input_from_file.cold.1"   ; string offset=681
	.asciz	"write_output_to_file.cold.1"   ; string offset=709
	.asciz	"main.cold.1"                   ; string offset=737
	.asciz	"OUTLINED_FUNCTION_0"           ; string offset=749
	.asciz	"_OUTLINED_FUNCTION_0"          ; string offset=769
	.asciz	"input"                         ; string offset=790
	.asciz	"input_size"                    ; string offset=796
	.asciz	"output_size"                   ; string offset=807
	.section	__DWARF,__debug_str_offs,regular,debug
	.long	0
	.long	47
	.long	102
	.long	158
	.long	173
	.long	227
	.long	232
	.long	273
	.long	280
	.long	299
	.long	302
	.long	316
	.long	319
	.long	323
	.long	326
	.long	333
	.long	339
	.long	345
	.long	356
	.long	362
	.long	349
	.long	368
	.long	377
	.long	385
	.long	392
	.long	398
	.long	436
	.long	426
	.long	411
	.long	404
	.long	446
	.long	453
	.long	457
	.long	464
	.long	473
	.long	477
	.long	483
	.long	489
	.long	493
	.long	502
	.long	291
	.long	286
	.long	510
	.long	517
	.long	524
	.long	531
	.long	539
	.long	544
	.long	565
	.long	574
	.long	588
	.long	581
	.long	600
	.long	605
	.long	610
	.long	617
	.long	647
	.long	631
	.long	624
	.long	661
	.long	669
	.long	674
	.long	252
	.long	612
	.long	681
	.long	709
	.long	737
	.long	769
	.long	749
	.long	790
	.long	796
	.long	807
	.section	__DWARF,__debug_addr,regular,debug
Lsection_info0:
.set Lset40, Ldebug_addr_end0-Ldebug_addr_start0 ; Length of contribution
	.long	Lset40
Ldebug_addr_start0:
	.short	5                               ; DWARF version number
	.byte	8                               ; Address size
	.byte	0                               ; Segment selector size
Laddr_table_base0:
	.quad	l_.str
	.quad	l_.str.1
	.quad	l_.str.2
	.quad	l_.str.3
	.quad	l_.str.4
	.quad	l_.str.5
	.quad	l_.str.6
	.quad	l_.str.7
	.quad	l_.str.8
	.quad	l_.str.9
	.quad	l_.str.10
	.quad	l_.str.12
	.quad	l_.str.14
	.quad	Lfunc_begin0
	.quad	Ltmp3
	.quad	Ltmp10
	.quad	Ltmp14
	.quad	Ltmp15
	.quad	Ltmp19
	.quad	Ltmp20
	.quad	Ltmp21
	.quad	Ltmp22
	.quad	Lfunc_begin1
	.quad	Ltmp27
	.quad	Ltmp25
	.quad	Ltmp32
	.quad	Ltmp36
	.quad	Ltmp38
	.quad	Lfunc_begin2
	.quad	Ltmp49
	.quad	Ltmp68
	.quad	Ltmp41
	.quad	Ltmp43
	.quad	Ltmp45
	.quad	Ltmp48
	.quad	Ltmp50
	.quad	Ltmp51
	.quad	Ltmp52
	.quad	Ltmp53
	.quad	Ltmp54
	.quad	Ltmp55
	.quad	Ltmp56
	.quad	Ltmp57
	.quad	Ltmp58
	.quad	Ltmp59
	.quad	Ltmp63
	.quad	Ltmp65
	.quad	Ltmp69
	.quad	Ltmp72
	.quad	Ltmp73
	.quad	Ltmp74
	.quad	Ltmp78
	.quad	Ltmp82
	.quad	Lfunc_begin3
	.quad	Lfunc_begin4
	.quad	Lfunc_begin5
	.quad	Ltmp87
	.quad	Lfunc_begin6
Ldebug_addr_end0:
	.section	__DWARF,__debug_names,regular,debug
Ldebug_names_begin:
.set Lset41, Lnames_end0-Lnames_start0  ; Header: unit length
	.long	Lset41
Lnames_start0:
	.short	5                               ; Header: version
	.short	0                               ; Header: padding
	.long	1                               ; Header: compilation unit count
	.long	0                               ; Header: local type unit count
	.long	0                               ; Header: foreign type unit count
	.long	12                              ; Header: bucket count
	.long	25                              ; Header: name count
.set Lset42, Lnames_abbrev_end0-Lnames_abbrev_start0 ; Header: abbreviation table size
	.long	Lset42
	.long	8                               ; Header: augmentation string size
	.ascii	"LLVM0700"                      ; Header: augmentation string
.set Lset43, Lcu_begin0-Lsection_info   ; Compilation unit 0
	.long	Lset43
	.long	0                               ; Bucket 0
	.long	1                               ; Bucket 1
	.long	8                               ; Bucket 2
	.long	9                               ; Bucket 3
	.long	12                              ; Bucket 4
	.long	0                               ; Bucket 5
	.long	13                              ; Bucket 6
	.long	0                               ; Bucket 7
	.long	15                              ; Bucket 8
	.long	19                              ; Bucket 9
	.long	22                              ; Bucket 10
	.long	23                              ; Bucket 11
	.long	274395349                       ; Hash in Bucket 1
	.long	691577533                       ; Hash in Bucket 1
	.long	2002163953                      ; Hash in Bucket 1
	.long	2090257189                      ; Hash in Bucket 1
	.long	-1921963995                     ; Hash in Bucket 1
	.long	-282664779                      ; Hash in Bucket 1
	.long	-143589579                      ; Hash in Bucket 1
	.long	2056321622                      ; Hash in Bucket 2
	.long	80989467                        ; Hash in Bucket 3
	.long	466678419                       ; Hash in Bucket 3
	.long	1950644907                      ; Hash in Bucket 3
	.long	-34160304                       ; Hash in Bucket 4
	.long	-371416606                      ; Hash in Bucket 6
	.long	-103762318                      ; Hash in Bucket 6
	.long	193495088                       ; Hash in Bucket 8
	.long	730786460                       ; Hash in Bucket 8
	.long	-1706460656                     ; Hash in Bucket 8
	.long	-104093792                      ; Hash in Bucket 8
	.long	1261628793                      ; Hash in Bucket 9
	.long	-1634592799                     ; Hash in Bucket 9
	.long	-154593283                      ; Hash in Bucket 9
	.long	2090499946                      ; Hash in Bucket 10
	.long	217009403                       ; Hash in Bucket 11
	.long	2090147939                      ; Hash in Bucket 11
	.long	-328142765                      ; Hash in Bucket 11
	.long	333                             ; String in Bucket 1: short
	.long	588                             ; String in Bucket 1: signed char
	.long	681                             ; String in Bucket 1: read_input_from_file.cold.1
	.long	286                             ; String in Bucket 1: FILE
	.long	436                             ; String in Bucket 1: long long
	.long	631                             ; String in Bucket 1: __darwin_size_t
	.long	411                             ; String in Bucket 1: __darwin_off_t
	.long	291                             ; String in Bucket 2: __sFILE
	.long	581                             ; String in Bucket 3: int8_t
	.long	624                             ; String in Bucket 3: size_t
	.long	426                             ; String in Bucket 3: __int64_t
	.long	404                             ; String in Bucket 4: fpos_t
	.long	252                             ; String in Bucket 6: read_input_from_file
	.long	647                             ; String in Bucket 6: unsigned long
	.long	319                             ; String in Bucket 8: int
	.long	769                             ; String in Bucket 8: _OUTLINED_FUNCTION_0
	.long	709                             ; String in Bucket 8: write_output_to_file.cold.1
	.long	302                             ; String in Bucket 8: unsigned char
	.long	737                             ; String in Bucket 9: main.cold.1
	.long	544                             ; String in Bucket 9: write_output_to_file
	.long	749                             ; String in Bucket 9: OUTLINED_FUNCTION_0
	.long	612                             ; String in Bucket 10: main
	.long	232                             ; String in Bucket 11: __ARRAY_SIZE_TYPE__
	.long	227                             ; String in Bucket 11: char
	.long	349                             ; String in Bucket 11: __sbuf
.set Lset44, Lnames7-Lnames_entries0    ; Offset in Bucket 1
	.long	Lset44
.set Lset45, Lnames15-Lnames_entries0   ; Offset in Bucket 1
	.long	Lset45
.set Lset46, Lnames20-Lnames_entries0   ; Offset in Bucket 1
	.long	Lset46
.set Lset47, Lnames3-Lnames_entries0    ; Offset in Bucket 1
	.long	Lset47
.set Lset48, Lnames12-Lnames_entries0   ; Offset in Bucket 1
	.long	Lset48
.set Lset49, Lnames18-Lnames_entries0   ; Offset in Bucket 1
	.long	Lset49
.set Lset50, Lnames10-Lnames_entries0   ; Offset in Bucket 1
	.long	Lset50
.set Lset51, Lnames4-Lnames_entries0    ; Offset in Bucket 2
	.long	Lset51
.set Lset52, Lnames14-Lnames_entries0   ; Offset in Bucket 3
	.long	Lset52
.set Lset53, Lnames17-Lnames_entries0   ; Offset in Bucket 3
	.long	Lset53
.set Lset54, Lnames11-Lnames_entries0   ; Offset in Bucket 3
	.long	Lset54
.set Lset55, Lnames9-Lnames_entries0    ; Offset in Bucket 4
	.long	Lset55
.set Lset56, Lnames2-Lnames_entries0    ; Offset in Bucket 6
	.long	Lset56
.set Lset57, Lnames19-Lnames_entries0   ; Offset in Bucket 6
	.long	Lset57
.set Lset58, Lnames6-Lnames_entries0    ; Offset in Bucket 8
	.long	Lset58
.set Lset59, Lnames24-Lnames_entries0   ; Offset in Bucket 8
	.long	Lset59
.set Lset60, Lnames21-Lnames_entries0   ; Offset in Bucket 8
	.long	Lset60
.set Lset61, Lnames5-Lnames_entries0    ; Offset in Bucket 8
	.long	Lset61
.set Lset62, Lnames22-Lnames_entries0   ; Offset in Bucket 9
	.long	Lset62
.set Lset63, Lnames13-Lnames_entries0   ; Offset in Bucket 9
	.long	Lset63
.set Lset64, Lnames23-Lnames_entries0   ; Offset in Bucket 9
	.long	Lset64
.set Lset65, Lnames16-Lnames_entries0   ; Offset in Bucket 10
	.long	Lset65
.set Lset66, Lnames1-Lnames_entries0    ; Offset in Bucket 11
	.long	Lset66
.set Lset67, Lnames0-Lnames_entries0    ; Offset in Bucket 11
	.long	Lset67
.set Lset68, Lnames8-Lnames_entries0    ; Offset in Bucket 11
	.long	Lset68
Lnames_abbrev_start0:
	.ascii	"\270\035"                      ; Abbrev code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	3                               ; DW_IDX_die_offset
	.byte	19                              ; DW_FORM_ref4
	.byte	4                               ; DW_IDX_parent
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; End of abbrev
	.byte	0                               ; End of abbrev
	.ascii	"\230$"                         ; Abbrev code
	.byte	36                              ; DW_TAG_base_type
	.byte	3                               ; DW_IDX_die_offset
	.byte	19                              ; DW_FORM_ref4
	.byte	4                               ; DW_IDX_parent
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; End of abbrev
	.byte	0                               ; End of abbrev
	.ascii	"\230."                         ; Abbrev code
	.byte	46                              ; DW_TAG_subprogram
	.byte	3                               ; DW_IDX_die_offset
	.byte	19                              ; DW_FORM_ref4
	.byte	4                               ; DW_IDX_parent
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; End of abbrev
	.byte	0                               ; End of abbrev
	.ascii	"\230\026"                      ; Abbrev code
	.byte	22                              ; DW_TAG_typedef
	.byte	3                               ; DW_IDX_die_offset
	.byte	19                              ; DW_FORM_ref4
	.byte	4                               ; DW_IDX_parent
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; End of abbrev
	.byte	0                               ; End of abbrev
	.ascii	"\230\023"                      ; Abbrev code
	.byte	19                              ; DW_TAG_structure_type
	.byte	3                               ; DW_IDX_die_offset
	.byte	19                              ; DW_FORM_ref4
	.byte	4                               ; DW_IDX_parent
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; End of abbrev
	.byte	0                               ; End of abbrev
	.byte	0                               ; End of abbrev list
Lnames_abbrev_end0:
Lnames_entries0:
Lnames7:
L13:
	.ascii	"\230$"                         ; Abbreviation code
	.long	722                             ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: short
Lnames15:
L7:
	.ascii	"\230$"                         ; Abbreviation code
	.long	1183                            ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: signed char
Lnames20:
L20:
	.ascii	"\230."                         ; Abbreviation code
	.long	1008                            ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: read_input_from_file.cold.1
Lnames3:
L15:
	.ascii	"\230\026"                      ; Abbreviation code
	.long	515                             ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: FILE
Lnames12:
L10:
	.ascii	"\230$"                         ; Abbreviation code
	.long	848                             ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: long long
Lnames18:
L12:
	.ascii	"\230\026"                      ; Abbreviation code
	.long	1616                            ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: __darwin_size_t
Lnames10:
L14:
	.ascii	"\230\026"                      ; Abbreviation code
	.long	832                             ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: __darwin_off_t
Lnames4:
L0:
	.ascii	"\230\023"                      ; Abbreviation code
	.long	523                             ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: __sFILE
Lnames14:
L22:
	.ascii	"\230\026"                      ; Abbreviation code
	.long	1175                            ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: int8_t
Lnames17:
L9:
	.ascii	"\230\026"                      ; Abbreviation code
	.long	1608                            ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: size_t
Lnames11:
L3:
	.ascii	"\230\026"                      ; Abbreviation code
	.long	840                             ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: __int64_t
Lnames9:
L5:
	.ascii	"\230\026"                      ; Abbreviation code
	.long	824                             ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: fpos_t
Lnames2:
L6:
	.ascii	"\230."                         ; Abbreviation code
	.long	341                             ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: read_input_from_file
Lnames19:
L25:
	.ascii	"\230$"                         ; Abbreviation code
	.long	1624                            ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: unsigned long
Lnames6:
L4:
	.ascii	"\230$"                         ; Abbreviation code
	.long	718                             ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: int
Lnames24:
L19:
	.ascii	"\230."                         ; Abbreviation code
	.long	1688                            ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: _OUTLINED_FUNCTION_0
Lnames21:
L11:
	.ascii	"\230."                         ; Abbreviation code
	.long	1113                            ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: write_output_to_file.cold.1
Lnames5:
L17:
	.ascii	"\230$"                         ; Abbreviation code
	.long	714                             ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: unsigned char
Lnames22:
L1:
	.ascii	"\230."                         ; Abbreviation code
	.long	1664                            ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: main.cold.1
Lnames13:
L2:
	.ascii	"\230."                         ; Abbreviation code
	.long	1018                            ; DW_IDX_die_offset
L16:                                    ; DW_IDX_parent
	.ascii	"\270\035"                      ; Abbreviation code
	.long	1260                            ; DW_IDX_die_offset
.set Lset69, L8-Lnames_entries0         ; DW_IDX_parent
	.long	Lset69
L21:
	.ascii	"\270\035"                      ; Abbreviation code
	.long	1674                            ; DW_IDX_die_offset
.set Lset70, L1-Lnames_entries0         ; DW_IDX_parent
	.long	Lset70
	.byte	0                               ; End of list: write_output_to_file
Lnames23:
	.ascii	"\230."                         ; Abbreviation code
	.long	1688                            ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: OUTLINED_FUNCTION_0
Lnames16:
L8:
	.ascii	"\230."                         ; Abbreviation code
	.long	1187                            ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: main
Lnames1:
L23:
	.ascii	"\230$"                         ; Abbreviation code
	.long	71                              ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: __ARRAY_SIZE_TYPE__
Lnames0:
L18:
	.ascii	"\230$"                         ; Abbreviation code
	.long	67                              ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: char
Lnames8:
L24:
	.ascii	"\230\023"                      ; Abbreviation code
	.long	726                             ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: __sbuf
	.p2align	2, 0x0
Lnames_end0:
.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
