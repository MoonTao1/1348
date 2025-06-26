from model_DINet import DINet
import torch
from utils.options import parser
args = parser.parse_args()
def build_model(args):
    model = None  # 默认返回 None
    if args.seq_len == 1:
        if args.network == 'salmae':
            model = DINet(512,False,512,False)
    if model is None:
        print("Error: Model is None")
    return model