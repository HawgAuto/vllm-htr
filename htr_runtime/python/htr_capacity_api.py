"""Native OpenAI server with object-preserving Qwen long-context overrides."""
import functools
import json
from htr_capacity_rope import apply_yarn


def prepare_args(argv=None):
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.launchers.cli_args import make_arg_parser,validate_parsed_serve_args
    parser=make_arg_parser(FlexibleArgumentParser(description='HTR long-context server'))
    args=parser.parse_args(argv);validate_parsed_serve_args(args)
    dtype=args.kv_cache_dtype
    factor={'int8_per_token_head':8,'int4_per_token_head':16}.get(dtype)
    if factor is None:raise ValueError('long-context entry requires released quantized KV')
    spec=args.speculative_config
    mtp=0 if spec is None else 3
    limit=2000000 if factor==8 else (3000000 if mtp else 4000000)
    if not 1048576<args.max_model_len<=limit:raise ValueError('context outside released KV/MTP range')
    if args.max_num_seqs!=1 or args.max_num_batched_tokens!=2048:raise ValueError('long-context profile requires one slot and chunk2048')
    rope=args.hf_overrides['text_config']['rope_parameters']
    if rope['rope_type']!='yarn' or rope['factor']!=factor:raise ValueError('unexpected requested YaRN settings')
    if spec is not None and (spec['method']!='mtp' or spec['num_speculative_tokens']!=3 or spec['max_model_len']!=args.max_model_len):raise ValueError('MTP configuration mismatch')
    args.hf_overrides=functools.partial(apply_yarn,factor=factor)
    print('HTR_CAPACITY_BOUND '+json.dumps({'context':args.max_model_len,'factor':factor,'mtp':mtp,'kv_cache_dtype':dtype}),flush=True)
    return args


def main():
    import uvloop
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup
    cli_env_setup()
    args=prepare_args()
    from vllm.entrypoints.launchers.api_server.entry import run_server
    uvloop.run(run_server(args))


if __name__=='__main__':main()
