ARG BASE=ghcr.io/hawgauto/haloloom-vllm-full-gfx1151@sha256:3bbe7f0425cfe698c27cae02c432acdad59e670f215db11a76cd15f3b53bc780
FROM ${BASE} AS native-build
USER root
COPY htr_runtime /opt/vllm-htr
RUN /opt/venv/bin/python3 /opt/vllm-htr/build_native.py --work /build/htr-native --jobs 2

FROM ${BASE}
USER root
COPY --from=native-build /opt/vllm-htr /opt/vllm-htr
COPY haloloom/qwen38/launch.py /opt/haloloom/qwen38/launch.py
COPY haloloom/qwen38/validate.py /opt/haloloom/qwen38/validate.py
COPY vllm/models/qwen4_exp/amd/model.py /opt/venv/lib/python3.14/site-packages/vllm/models/qwen4_exp/amd/model.py
COPY vllm/models/qwen4_exp/amd/ple_layer.py /opt/venv/lib/python3.14/site-packages/vllm/models/qwen4_exp/amd/ple_layer.py
COPY vllm/models/qwen4_exp/amd/model_state.py /opt/venv/lib/python3.14/site-packages/vllm/models/qwen4_exp/amd/model_state.py
COPY vllm/models/qwen4_exp/amd/qsa.py /opt/venv/lib/python3.14/site-packages/vllm/models/qwen4_exp/amd/qsa.py
RUN /opt/venv/bin/python3 /opt/vllm-htr/install_layer.py
ENV PYTHONPATH=/opt/vllm-htr/python PYTHONDONTWRITEBYTECODE=1 HTR_NATIVE_DIR=/opt/vllm-htr/native-bin
USER 1000:1000
WORKDIR /workspace
ENTRYPOINT ["/opt/venv/bin/python3", "/opt/vllm-htr/launch_htr.py"]
