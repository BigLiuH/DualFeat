import argparse


def parameter_parser():

    parser = argparse.ArgumentParser(description="Run MMGCN.")

    parser.add_argument("--dataset-path",
                        nargs="?",
                        default="../../datasets",
                        help="Training datasets.")

    parser.add_argument("--epoch",#embdding  模块的epoch数
                        type=int,
                        default=800,
                        help="Number of training epochs. Default is 651.")

    parser.add_argument("--gcn-layers",# GCN模块的层数
                        type=int,
                        default=2,
                        help="Number of Graph Convolutional Layers. Default is 2.")

    parser.add_argument("--out-channels", #这个是ECA模块的输出维度
                        type=int,
                        default=128,
                        help="out-channels of cnn. Default is 128.")

    parser.add_argument("--miRNA-number", #RNA的总数
                        type=int,
                        default=924,
                        help="miRNA number. Default is 585.")

    parser.add_argument("--fm",          #GAT模块的输出维度
                        type=int,
                        default=256,
                        help="miRNA feature dimensions. Default is 256.")

    parser.add_argument("--disease-number",#miRNA的数量
                        type=int,
                        default=420,
                        help="disease number. Default is 88.")

    parser.add_argument("--fd",#miRNA在GAT模块的输出维度
                        type=int,
                        default=256,
                        help="disease feature number. Default is 256.")

    parser.add_argument("--view",
                        type=int,
                        default=2,
                        help="views number. Default is 2(2 datasets for miRNA and disease sim)")


    parser.add_argument("--validation",
                        type=int,
                        default=5,
                        help="5 cross-validation.")


    return parser.parse_args()