"""Public callable HF override: preserve Qwen config objects and MTP inheritance."""
import copy,json,os
TRAINED_CONTEXT=262144

def apply_yarn(config,*,factor):
    if factor not in (8,16):raise ValueError('only the approved YaRN 8/16 capacity profiles are permitted')
    # Native config_parser invokes callables on a type-discovery sentinel first.
    if str(config.model_type).startswith('dummy_'):return config
    if config.model_type not in ('qwen4_exp','qwen4_exp_text','qwen4_exp_mtp'):raise ValueError('unexpected model type for capacity override')
    text=config.get_text_config()
    if text.num_hidden_layers!=48 or text.hidden_size!=2560:raise ValueError('unexpected model geometry')
    rope=copy.deepcopy(text.rope_parameters)
    if rope.get('rope_type') not in ('default','yarn'):raise ValueError('unexpected source RoPE')
    rope.update(rope_type='yarn',factor=float(factor),original_max_position_embeddings=TRAINED_CONTEXT)
    for obj in (config,text):
        obj.max_position_embeddings=TRAINED_CONTEXT*factor
        obj.original_max_position_embeddings=TRAINED_CONTEXT
        obj.rope_parameters=copy.deepcopy(rope)
        obj.rope_scaling=copy.deepcopy(rope)
    print('CAPACITY_YARN_APPLIED '+json.dumps({'pid':os.getpid(),'model_type':config.model_type,'architectures':config.architectures,'max_position_embeddings':text.max_position_embeddings,'original_max_position_embeddings':TRAINED_CONTEXT,'rope_parameters':text.rope_parameters}),flush=True)
    return config
