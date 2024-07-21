
VOCAB_FILE = "model/prot_t5_xl_half_uniref50-enc/vocab.txt"

log_path = "output/logs"
plot_path = "output/images/loss_acc.png"
data_dir = "data/"
cache_dir = "model/"

model = "model/prot_t5_xl_half_uniref50-enc"

num_train_epochs = 25
is_train = False
is_valid = True
is_single = True
if is_single == True:
    is_train = False
    is_valid = False
    num_train_epochs = 1
if is_valid == True and is_train == False:
    num_train_epochs = 1

# seq_name = "6DLM_A"
seq = "MKELFEVIFEGVNTSRLFFLLKEIESKSDRIFDFNFSEDFFSSNVNVFSELLIDSFLGFNGDLYFGVSMEGFSVKDGLKLPVVLLRVLKYEGGVDVGLCFYMNDFNSAGKVMLEFQKYMNGISADFGFENFYGGLEPASDQETRFFTNNRLGPLL"

TASK_NAMES = ['pssp_3', 'pssp_8']
task_name = "pssp_3"

TRAIN_DATASET = ['cullpdb', 'Netsurfp2']
train_dataset = "Netsurfp2"
TEST_DATASET = ['cb513', 'casp12', 'ts115', 'new364','casp11']
test_dataset = "casp12"


fasta_file = f'data/rcsb_pdb_1A1X.fasta'

flag_words = ["[PAD]", "[CLP]", "[SEP]", "[UNK]"]
do_lower_case = True
train_batch_size = 16
eval_batch_size = 1
learning_rate = 2e-5

warmup_proportion = 0.1
no_cuda = False
seed = 47
gradient_accumulation_steps = 1
fp16 = False
loss_scale = 0.
embds = "t5"    #  esm t5

if train_dataset == 'cullpdb':
    max_seq_length = 700
elif train_dataset == 'Netsurfp2':
    max_seq_length = 1632

if test_dataset == 'cb513':
    max_valid_length = 880
elif test_dataset == 'casp12':
    max_valid_length = 1500
elif test_dataset == 'ts115':
    max_valid_length = 1112
elif test_dataset == 'new364':
    max_valid_length = 1590
elif test_dataset == 'casp11':
    max_valid_length = 630

if task_name == 'pssp_3':
    labels = ['C', 'H', 'E']
    types = ['0', '1', '2']
    num_tags = 3
    output_dir = "output/checkpoint/Q3"
    npz_path = f't5/{train_dataset}.npz'
    model_path = f'output/checkpoint/model_pssp3.bin'
elif task_name == 'pssp_8':
    labels = ['C', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
    types = ['0', '1', '2','3','4','5','6','7']
    num_tags = 8
    output_dir = "output/checkpoint/Q8"
    npz_path = f't5/{train_dataset}.npz'
    model_path = f'output/checkpoint/model_pssp8.bin'

pt_size = 1024
encode_size = 512   #best:512

if embds == "esm":
    hidden_size = 1280
else:
    hidden_size = pt_size + encode_size

device = "cuda:0"

# TRAIN_PATH = f"data/{task_name}_data/{task_name}.train.json"
# VALID_PATH = f"data/{task_name}_data/{task_name}.dev.json"
TRAIN_PATH = f"data/{task_name}_data/{train_dataset}.json"
VALID_PATH = f"data/{task_name}_data/{test_dataset}.json"

def count_lines(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return sum(1 for _ in file)
filename = 'model/prot_t5_xl_half_uniref50-enc/vocab.txt'
vocab_count = count_lines(filename)
