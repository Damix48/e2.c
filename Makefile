# e2 - EmbeddingGemma-2 int8 inference engine
CC      ?= cc
CSTD    := -std=gnu11
WARN    := -Wall -Wextra -Wno-unused-parameter -Wno-shift-negative-value
OPT     := -O3 -ffast-math -fno-finite-math-only -funroll-loops
CFLAGS  ?= $(CSTD) $(WARN) $(OPT) -I.
LDFLAGS ?= -lm -lpthread
DOTPROD ?= 0
ifeq ($(DOTPROD),1)
CFLAGS  += -march=armv8.2-a+dotprod
endif

# Optional system libraries. By default they are auto-detected; force with
# USE_SYSTEM_IMAGE=0/1, USE_WEBP=0/1, USE_SNDFILE=0/1 on the make line.
HAVE_PNG     := $(shell pkg-config --exists libpng 2>/dev/null && echo 1)
HAVE_JPEG    := $(shell pkg-config --exists libjpeg 2>/dev/null && echo 1)
HAVE_WEBP    := $(shell pkg-config --exists libwebp 2>/dev/null && echo 1)
HAVE_SNDFILE := $(shell pkg-config --exists sndfile 2>/dev/null && echo 1)
ifeq ($(origin USE_SYSTEM_IMAGE),undefined)
USE_SYSTEM_IMAGE := $(if $(and $(HAVE_PNG),$(HAVE_JPEG)),1,0)
endif
ifeq ($(origin USE_WEBP),undefined)
USE_WEBP := $(HAVE_WEBP)
endif
ifeq ($(origin USE_SNDFILE),undefined)
USE_SNDFILE := $(HAVE_SNDFILE)
endif
ifeq ($(USE_SYSTEM_IMAGE),1)
CFLAGS  += -DE2_USE_SYSTEM_IMAGE $(shell pkg-config --cflags libpng libjpeg)
LDFLAGS += $(shell pkg-config --libs libpng libjpeg)
endif
ifeq ($(USE_WEBP),1)
CFLAGS  += -DE2_USE_WEBP $(shell pkg-config --cflags libwebp)
LDFLAGS += $(shell pkg-config --libs libwebp)
endif
ifeq ($(USE_SNDFILE),1)
CFLAGS  += -DE2_USE_SNDFILE $(shell pkg-config --cflags sndfile)
LDFLAGS += $(shell pkg-config --libs sndfile)
endif

SRC := src/e2.c src/e2_model.c src/e2_util.c src/e2_gemm.c src/e2_ops.c \
       src/e2_attn.c src/e2_text.c src/e2_tokenizer.c src/e2_image.c \
       src/e2_vision.c src/e2_audio.c src/e2_audio_file.c src/e2_multimodal.c \
       src/e2_gpu.c src/e2_gpu_ops.c src/e2_gpu_l0.c
OBJ := $(SRC:.c=.o)
HDR := e2.h src/e2_internal.h src/e2_gpu.h src/e2_gpu_int.h src/e2_gpu_l0.h src/e2_gpu_spv.h

.PHONY: all clean test bench compare pack help spv

all: libe2.a e2cli
	@echo "e2: system-image=$(USE_SYSTEM_IMAGE) webp=$(USE_WEBP) sndfile=$(USE_SNDFILE)"

help:
	@echo "make            build libe2.a + e2cli"
	@echo "make test       run C self-checks (uses models/)"
	@echo "make bench      text batch benchmark"
	@echo "make compare    e2 vs OpenVINO (f32/int8), all tasks (needs .venv)"
	@echo "make pack       quantize all six model variants (needs .venv)"
	@echo "make DOTPROD=1  build with ARMv8.2 dot-product instructions"

libe2.a: $(OBJ)
	ar rcs $@ $^

%.o: %.c $(HDR)
	$(CC) $(CFLAGS) -c -o $@ $<

e2cli: tools/e2cli.c libe2.a $(HDR)
	$(CC) $(CFLAGS) -o $@ $< libe2.a $(LDFLAGS)

tests/e2_test: tests/e2_test.c libe2.a $(HDR)
	$(CC) $(CFLAGS) -o $@ $< libe2.a $(LDFLAGS)

tests/e2_bench: tests/e2_bench.c libe2.a $(HDR)
	$(CC) $(CFLAGS) -o $@ $< libe2.a $(LDFLAGS)

tests/e2_bench_full: tests/e2_bench_full.c libe2.a $(HDR)
	$(CC) $(CFLAGS) -o $@ $< libe2.a $(LDFLAGS)

# regenerate src/e2_gpu_spv.h from tools/gpu/*.cl (needs clang + xxd; the
# generated header is checked in so a normal build does not)
spv:
	sh tools/e2_spv.sh

# dev-only Level Zero GEMM probe (not shipped; dlopens libze_loader at runtime)
gpu-probe: tools/gpu_probe.c src/e2_gpu_l0.c libe2.a $(HDR) src/e2_gpu_l0.h src/e2_gpu_spv.h
	$(CC) $(CFLAGS) -o $@ tools/gpu_probe.c src/e2_gpu_l0.c libe2.a $(LDFLAGS)

# dev-only device-GEMM correctness check (not shipped)
gpu-check: tools/gpu_gemm_check.c libe2.a $(HDR)
	$(CC) $(CFLAGS) -o $@ $< libe2.a $(LDFLAGS)

# dev-only kernel microbenchmark (not shipped)
mb: tools/mb.c libe2.a $(HDR)
	$(CC) $(CFLAGS) -o $@ $< libe2.a $(LDFLAGS)

test: tests/e2_test
	./tests/e2_test models

bench: tests/e2_bench
	./tests/e2_bench models

bench-full: tests/e2_bench_full
	. .venv/bin/activate && python tools/bench_compare.py --sources e2

compare: tests/e2_bench_full
	. .venv/bin/activate && python tools/bench_compare.py

pack:
	. .venv/bin/activate && \
	python tools/e2_pack.py --modalities text --out models/e2-text.bin --tokenizer-out models/e2-tokenizer.bin && \
	python tools/e2_pack.py --modalities text,image --out models/e2-text+image.bin --tokenizer-out models/e2-tokenizer.bin && \
	python tools/e2_pack.py --modalities text,audio --out models/e2-text+audio.bin --tokenizer-out models/e2-tokenizer.bin && \
	python tools/e2_pack.py --modalities text,video --out models/e2-text+video.bin --tokenizer-out models/e2-tokenizer.bin && \
	python tools/e2_pack.py --modalities text,image,audio --out models/e2-text+image+audio.bin --tokenizer-out models/e2-tokenizer.bin && \
	python tools/e2_pack.py --modalities text,image,audio,video --out models/e2-text+image+audio+video.bin --tokenizer-out models/e2-tokenizer.bin

clean:
	rm -f $(OBJ) libe2.a e2cli tests/e2_test tests/e2_bench tests/e2_bench_full mb gpu-probe gpu-check
