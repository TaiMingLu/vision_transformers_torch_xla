"""Minimal model registry surface.

Only the vision transformer variants required by our training entrypoints are
imported eagerly. Additional architectures can be re-enabled by importing their
modules explicitly before calling create_model.
"""

from __future__ import annotations

# Import the handful of model definitions we depend on.
from .my_vit import *  # noqa: F401,F403
from .vision_transformer import *  # noqa: F401,F403

# Re-export the helper APIs that downstream code expects to find here.
from ._builder import (  # noqa: F401
    build_model_with_cfg,
    load_pretrained,
    load_custom_pretrained,
    resolve_pretrained_cfg,
    set_pretrained_download_progress,
    set_pretrained_check_hash,
)
from ._factory import (  # noqa: F401
    create_model,
    parse_model_name,
    safe_model_name,
)
from ._features import (  # noqa: F401
    FeatureInfo,
    FeatureHooks,
    FeatureHookNet,
    FeatureListNet,
    FeatureDictNet,
)
from ._features_fx import (  # noqa: F401
    FeatureGraphNet,
    GraphExtractNet,
    create_feature_extractor,
    get_graph_node_names,
    register_notrace_module,
    is_notrace_module,
    get_notrace_modules,
    register_notrace_function,
    is_notrace_function,
    get_notrace_functions,
)
from ._helpers import (  # noqa: F401
    clean_state_dict,
    load_state_dict,
    load_checkpoint,
    remap_state_dict,
    resume_checkpoint,
)
from ._hub import (  # noqa: F401
    load_model_config_from_hf,
    load_state_dict_from_hf,
    push_to_hf_hub,
    save_for_hf,
)
from ._manipulate import (  # noqa: F401
    model_parameters,
    named_apply,
    named_modules,
    named_modules_with_params,
    group_modules,
    group_parameters,
    checkpoint_seq,
    checkpoint,
    adapt_input_conv,
)
from ._pretrained import (  # noqa: F401
    PretrainedCfg,
    DefaultCfg,
    filter_pretrained_cfg,
)
from ._prune import adapt_model_from_string  # noqa: F401
from ._registry import (  # noqa: F401
    split_model_name_tag,
    get_arch_name,
    generate_default_cfgs,
    register_model,
    register_model_deprecations,
    model_entrypoint,
    list_models,
    list_pretrained,
    get_deprecated_models,
    is_model,
    list_modules,
    is_model_in_modules,
    is_model_pretrained,
    get_pretrained_cfg,
    get_pretrained_cfg_value,
    get_arch_pretrained_cfgs,
)
