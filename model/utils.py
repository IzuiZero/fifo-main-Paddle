try:
    from paddle.hub import load_state_dict_from_url
except ImportError:
    from paddle.utils.model_zoo import load_url as load_state_dict_from_url