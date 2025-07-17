from .method import UlvaFormer

__all__ = ['UlvaFormer','get_model']

def get_model(args):
    return UlvaFormer(
        d_model=int(args.d_model),
        n_heads=args.n_heads,
        n_layer=args.n_layer,
        r_forward=args.r_forward,
        dropout=args.dropout,
        drop_path=args.drop_path,
        data_back=args.data_back,
        data_pred=args.data_pred,
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        attn_bias=False,
    )