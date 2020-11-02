# This script runs S2S-LHS and S2S-LHS+PA{pa_index}{+Type} models.

from typing import *
from seutil import IOUtils
from shared import *

import numpy as np
import argparse
import os
import yaml


parser=argparse.ArgumentParser()
parser.add_argument('-mode', '--mode', required=True,
                    default='train',
                    type=str, help='mode for running codes (train|test|testval|assemble)')
parser.add_argument('-save_dir', '--save_dir', required=False,
                    default='',
                    type=str, help='save directory path')
parser.add_argument('-feat', '--feat', required=False,
                    default='', choices=["l", "lpa", "lpa+typeappend"],
                    type=str, help='type of model')
parser.add_argument('-pa_index', '--pa_index',
                    default=1, required=False,
                    type=str, help='which previous assignement is used')
parser.add_argument('-force_delete', '--force_delete', required=False,
                    action='store_true', help='flag to force delete for training mode')
parser.add_argument('-config', '--config', required=False,
                    default='./config/ex_ms_config_default.yml',
                    type=str, help='config file path')
parser.add_argument('-which', '--which', required=False,
                    default=[],
                    type=str, nargs='+', help='model list to be assembled')
parser.add_argument('-stop_after_conversion', '--stop-after-conversion', required=False,
                    default=False,
                    type=str, help='stop the script after conversion to txt is done')


def get_src_types(config_dict):
    assert config_dict['feat'] in ['l', 'lpa', 'lpa+typeappend']

    if config_dict['feat'] == 'lpa':
        config_dict['src_types'] = ["l", "prevassign"]
    elif config_dict['feat'] == 'lpa+typeappend':
        config_dict['src_types'] = ["l", "type", "prevassign", "patype"]
    else:
        config_dict['src_types'] = ["l"]
    return config_dict


def read_yaml(yaml_path):
    assert os.path.exists(yaml_path)
    with open(yaml_path, 'r') as f:
        try:
            config_dict = yaml.safe_load(f)
            return config_dict
        except yaml.YAMLError as exc:
            print(exc)


def write_yaml(datadict, yaml_path):
    with open(yaml_path, 'w') as f:
        yaml.dump(datadict, f)


def check_config(config_dict, args):
    if args.feat != '':
        config_dict["feat"] = args.feat
        if args.feat == "lpa+typeappend":
            config_dict["type_append"] = True

    if "lpa" == args.feat[:3]:
        config_dict["augment"] = args.pa_index

    config_dict = get_src_types(config_dict)
    if args.save_dir=='':
        config_dict["save_dir"] = SAVEDIR
    else:
        config_dict["save_dir"] = args.save_dir
    config_dict["force_delete"] = args.force_delete
    if args.mode.lower() == "train":
        if config_dict["force_delete"]:
            os.system('rm -rf '+os.path.join(DATADIR, config_dict["save_dir"]))
            os.system('rm -rf '+os.path.join(CONFIGDIR, config_dict["save_dir"]))
            os.system('rm -rf '+os.path.join(MODELDIR, config_dict["save_dir"]))
            os.system('rm -rf '+os.path.join(LOGDIR, config_dict["save_dir"]))
            os.system('rm -rf '+os.path.join(TESTDIR, config_dict["save_dir"]))
        #end if
        os.system('mkdir -p '+os.path.join(DATADIR, config_dict["save_dir"]))
        os.system('mkdir -p '+os.path.join(CONFIGDIR, config_dict["save_dir"]))
        os.system('mkdir -p '+os.path.join(MODELDIR, config_dict["save_dir"]))
        os.system('mkdir -p '+os.path.join(LOGDIR, config_dict["save_dir"]))
        os.system('mkdir -p '+os.path.join(TESTDIR, config_dict["save_dir"]))
    #end if
    
    config_dict["preprocess_config_default"] = os.path.join(CONFIGDIR, config_dict["preprocess_config"])
    config_dict["train_config_default"] = os.path.join(CONFIGDIR, config_dict["train_config"])
    config_dict["translate_config_default"] = os.path.join(CONFIGDIR, config_dict["translate_config"])
    config_dict["preprocess_config"] = os.path.join(CONFIGDIR, config_dict["save_dir"], config_dict["preprocess_config"])
    config_dict["train_config"] = os.path.join(CONFIGDIR, config_dict["save_dir"], config_dict["train_config"])
    config_dict["translate_config"] = os.path.join(CONFIGDIR, config_dict["save_dir"], config_dict["translate_config"])
    st_str = ".subtokenize"
    testmode_str = ".val" if args.mode=="testval" else ""
    config_dict["save_data"] = os.path.join(DATADIR, config_dict["save_dir"],f"assignments{st_str}")
    config_dict["model_name"] = f"assignments{st_str}.lattn.model"
    config_dict["pt_file"] = config_dict["model_name"]+"_step_*.pt"
    config_dict["train_log"] = f"assignments{st_str}.lattn.model.log"
    config_dict["pred_file"] = f"pred.assignments{st_str}.lattn.model.txt"
    config_dict["pred_log"] = f"pred.assignments{st_str}.lattn.model.log"
    config_dict["pred_print_file"] = f"total.assignments{st_str}.lattn.model.txt"
    config_dict["metric_log"] = f"testlog.assignments{st_str}.lattn.model.log"
    config_dict["src_train_file"] = f"src.train{st_str}.txt"
    config_dict["tgt_train_file"] = f"tgt.train{st_str}.txt"
    config_dict["src_val_file"] = f"src.val{st_str}.txt"
    config_dict["tgt_val_file"] = f"tgt.val{st_str}.txt"
    config_dict["src_test_file"] = f"src.test{st_str}.txt"
    config_dict["tgt_test_file"] = f"tgt.test{st_str}.txt"
    return config_dict


def config_parse(args):
    config_dict = read_yaml(args.config)
    config_dict = check_config(config_dict, args)
    if args.mode.lower() == "train":
        # Save configuration for reference/debugging
        yaml_path = os.path.join(CONFIGDIR, config_dict["save_dir"], "config.yml")
        write_yaml(config_dict, yaml_path)
    else:
        config_dict["mode"] = args.mode.lower()
        if args.mode.lower() == 'test' or args.mode.lower() == "testval":
            assert args.save_dir != ''
    return config_dict


def preprocess_parse_args(config_dict):
    pp_config_dict = read_yaml(config_dict["preprocess_config_default"])
    pp_config_dict["train_src"] = config_dict["save_data"]
    pp_config_dict["train_tgt"] = os.path.join(DATADIR, config_dict["save_dir"], config_dict["tgt_train_file"])
    pp_config_dict["valid_src"] = os.path.join(DATADIR, config_dict["save_dir"], config_dict["src_val_file"])
    pp_config_dict["valid_tgt"] = os.path.join(DATADIR, config_dict["save_dir"], config_dict["tgt_val_file"])
    pp_config_dict["save_data"] = config_dict["save_data"]
    pp_config_dict["src_types"] = config_dict["src_types"]
    write_yaml(pp_config_dict, config_dict["preprocess_config"])
    return


def train_parse_args(config_dict):
    tr_config_dict = read_yaml(config_dict["train_config_default"])
    tr_config_dict["data"] = config_dict["save_data"]
    tr_config_dict["save_model"] = os.path.join(MODELDIR, config_dict["save_dir"], config_dict["model_name"])
    tr_config_dict["log_file"] = os.path.join(LOGDIR, config_dict["save_dir"], config_dict["train_log"])
    tr_config_dict["src_types"] = config_dict["src_types"]
    if "type_append" in config_dict.keys():
        tr_config_dict["type_append"] = config_dict["type_append"]
    write_yaml(tr_config_dict, config_dict["train_config"])
    return


def translate_parse_args(config_dict):
    tl_config_dict = read_yaml(config_dict["translate_config_default"])
    tl_config_dict["model"] = get_most_recent_model(config_dict)
    if "type_append" in config_dict.keys():
        tl_config_dict["type_append"] = config_dict["type_append"]

    if config_dict["mode"]=="testval":
        config_dict["pred_file"] = "pred.val."+".".join(config_dict["pred_file"].split(".")[1:])
        config_dict["pred_log"] = "pred.val."+".".join(config_dict["pred_log"].split(".")[1:])
        config_dict["pred_print_file"] = "total.val."+".".join(config_dict["pred_print_file"].split(".")[1:])
        config_dict["metric_log"] = "testlog.val."+".".join(config_dict["metric_log"].split(".")[1:])

    tl_config_dict["data_mode"] = "val" if config_dict["mode"]=="testval" else "test"
    tl_config_dict["src"] = os.path.join(DATADIR, config_dict["save_dir"], config_dict["src_test_file"])
    tl_config_dict["tgt"] = os.path.join(DATADIR, config_dict["save_dir"], config_dict["tgt_test_file"])
    tl_config_dict["output"] = os.path.join(TESTDIR, config_dict["save_dir"], config_dict["pred_file"])
    tl_config_dict["log_file"] = os.path.join(LOGDIR, config_dict["save_dir"], config_dict["pred_log"])
    tl_config_dict["src_types"] = config_dict["src_types"]
    write_yaml(tl_config_dict, config_dict["translate_config"])
    return

    
def get_most_recent_model(config_dict):
    import glob
    file_list = glob.glob(os.path.join(MODELDIR, config_dict["save_dir"], config_dict["pt_file"]))
    if not file_list:
        raise("No pre-trained model is found!")
    elif len(file_list) == 1:
        return file_list[0]
    return sorted(file_list, key=lambda name:int(name.split("step_")[1].split(".pt")[0]))[-1]


def convert_json2txt(config_dict, data_types: List[str] = None):
    if data_types is None:
        data_types = ["train", "val", "test"]

    for data_type in data_types:
        data_list = IOUtils.load(os.path.join(DATADIR, config_dict["intermediate_data_dir"], f"{data_type}.json"), IOUtils.Format.json)

        for src_type in config_dict["src_types"]:
            output_path = os.path.join(DATADIR, config_dict["save_dir"], f"src.{src_type}.{data_type}.txt")
            pa_i = int(config_dict["augment"])
            if src_type == "l":
                field = "l"
            elif src_type == "type":
                field = "l-type"
            elif src_type == "prevassign":
                field = f"pa{pa_i}"
            elif src_type == "patype":
                field = f"pa{pa_i}-type"
            else:
                raise ValueError(f"Unknown src_type {src_type}")
            # end if

            with open(output_path, "w") as f:
                for data in data_list:
                    if len(data[field]) == 0:
                        if field.endswith("-type"):
                            f.write("<pad>\n")
                        else:
                            f.write("<empty>\n")
                        # end if
                    else:
                        f.write(data[field]+"\n")
                    # end if
                # end for
            # end with
        # end for

        fn_output_path = os.path.join(DATADIR, config_dict["save_dir"], f"src.fn.{data_type}.txt")
        IOUtils.dump(fn_output_path, "".join([data["file_sha"]+"\n" for data in data_list]), IOUtils.Format.txt)

        tgt_output_path = os.path.join(DATADIR, config_dict["save_dir"], f"tgt.{data_type}.txt")
        # [3:-2]: remove prefix "<= " and suffix " ;"
        IOUtils.dump(tgt_output_path, "".join([data["r"][3:-2]+"\n" for data in data_list]), IOUtils.Format.txt)
    # end for
    print("Conversion into txt is done.")
    return


def get_test_bleuacc(config_dict):
    dpath = os.path.join(DATADIR, config_dict["save_dir"])
    ppath = os.path.join(TESTDIR, config_dict["save_dir"], config_dict["pred_file"])
    tgt_mode = "val" if config_dict["mode"]=="testval" else "test"
    tpath = os.path.join(DATADIR, config_dict["save_dir"], f"tgt.{tgt_mode}.txt")
    rpath = os.path.join(TESTDIR, config_dict["save_dir"], config_dict["metric_log"])
    os.system(f"python measure_bleuacc.py -d {dpath} -p {ppath} -t {tpath} -r {rpath}")
    return
    

def copy_assemble_data_init(config_dict, model, assemble_model_name):
    from shutil import copyfile
    
    if not os.path.exists(os.path.join(TESTDIR, assemble_model_name)):
        os.mkdir(os.path.join(TESTDIR, assemble_model_name))

    for mode in ["val", "test"]:
        # Src lhs
        copyfile(os.path.join(DATADIR, model, f"src.l.{mode}.txt"),
                 os.path.join(TESTDIR, assemble_model_name, f"lhs.{mode}.txt"))
    # end for

    # Target for val
    copyfile(os.path.join(DATADIR, model, f"tgt.val.txt"),
             os.path.join(TESTDIR, assemble_model_name, f"tgt.val.txt"))
    return
    

def copy_assemble_data(config_dict, model, assemble_model_name):
    from shutil import copyfile

    for mode in ["val", "test"]:
        # Predictions
        copyfile(os.path.join(TESTDIR, model, "pred.assignments.subtokenize.lattn.model.txt" if mode == "test" else "pred.val.assignments.subtokenize.lattn.model.txt"),
                 os.path.join(TESTDIR, assemble_model_name, f"pred.{model}.{mode}.txt"))

        # Logprobs
        copyfile(os.path.join(TESTDIR, model, "pred-logprobs.assignments.subtokenize.lattn.model.txt" if mode == "test" else "pred-logprobs.val.assignments.subtokenize.lattn.model.txt"),
                 os.path.join(TESTDIR, assemble_model_name, f"logprob.{model}.{mode}.txt"))

        # Prev assign
        copyfile(os.path.join(DATADIR, model, f"src.prevassign.{mode}.txt"),
                 os.path.join(TESTDIR, assemble_model_name, f"prevassign.{model}.{mode}.txt"))
    # end for
    return


def load_data(f) -> List[List[str]]:
    return [l.split() for l in IOUtils.load(f, IOUtils.Format.txt).splitlines()]


def load_logprob(f) -> List[float]:
    return [float(l) for l in IOUtils.load(f, IOUtils.Format.txt).splitlines()]


def get_similarity(seq1, seq2):
    # 1 - jaccard_distance
    if len(seq1) == len(seq2) == 0:  return 0
    label1 = set(seq1)
    label2 = set(seq2)
    return len(label1.intersection(label2)) / len(label1.union(label2))


def get_accuracy(target: List[str], pred: List[str]) -> float:
    total = max(len(target), len(pred))
    if total == 0:  return 0
    correct = 0
    for j in range(min(len(target), len(pred))):
        if target[j] == pred[j]:  correct += 1
    # end for
    return correct / total


def get_prevassign_rhs(prevassign: List[str]) -> List[str]:
    if len(prevassign) > 1:
        return prevassign[prevassign.index("<=")+1:-1]
    else:
        return []
    # end if


def get_prevassign_lhs(prevassign: List[str]) -> List[str]:
    if len(prevassign) > 1:
        return prevassign[:prevassign.index("<=")]
    else:
        return []
    # end if


def get_features(models, num_data, preds, logprobs, prevassigns, src):
    features = list()
    for j in range(num_data):
        row = list()

        # src length
        row.append(len(src[j]))

        # similarities between src with prevassign[k] lhs
        lhs_similarities = list()
        for k in models:
            pa = prevassigns[k][j]
            pa_lhs = get_prevassign_lhs(pa)
            lhs_similarities.append(get_similarity(src[j], pa_lhs))
        # end for
        row.extend(lhs_similarities)
        row.append(max(lhs_similarities))
        row.append(np.mean(lhs_similarities))
        row.append(min(lhs_similarities))

        for i in models:
            # pred length
            row.append(len(preds[i][j]))

            # has prevassign[i]
            row.append(1 if len(prevassigns[i][j]) > 1 else 0)

            # prevassign[i] rhs length
            row.append(len(get_prevassign_rhs(prevassigns[i][j])))

            # model prob
            row.append(np.exp(logprobs[i][j]))

            # similarities between pred with prevassign[k] rhs
            rhs_similarities = list()
            rhs_accs = list()
            for k in models:
                pa = prevassigns[k][j]
                pa_rhs = get_prevassign_rhs(pa)
                rhs_similarities.append(get_similarity(preds[i][j], pa_rhs))
            # end for
            row.extend(rhs_similarities)
            row.append(max(rhs_similarities))
            row.append(np.mean(rhs_similarities))
            row.append(min(rhs_similarities))

            # similarities between pred with src
            row.append(get_similarity(preds[i][j], src[j]))

            # similarities between prevassign[i] rhs with prevassign[k] rhs
            pa_lhs_similarities = list()
            pa_rhs_similarities = list()
            for k in models:
                pa_i = prevassigns[i][j]
                pa_k = prevassigns[k][j]
                pa_lhs_similarities.append(get_similarity(get_prevassign_lhs(pa_i), get_prevassign_lhs(pa_k)))
                pa_rhs_similarities.append(get_similarity(get_prevassign_rhs(pa_i), get_prevassign_rhs(pa_k)))
            # end for
            row.extend(pa_lhs_similarities)
            row.append(max(pa_lhs_similarities))
            row.append(np.mean(pa_lhs_similarities))
            row.append(min(pa_lhs_similarities))
            row.extend(pa_rhs_similarities)
            row.append(max(pa_rhs_similarities))
            row.append(np.mean(pa_rhs_similarities))
            row.append(min(pa_rhs_similarities))
        # end for
        # print(row)
        features.append(row)
    # end for
    return np.array(features)


def get_targets(models, num_data, preds, logprobs, prevassigns, src, tgt):
    targets = list()
    for j in range(num_data):
        accs = list()
        # For each model: acc_pred, acc_copy_from_prevassign
        for model in models:
            accs.append(get_accuracy(preds[model][j], tgt[j]))
            accs.append(get_accuracy(get_prevassign_rhs(prevassigns[model][j]), tgt[j]))
        # end for
        targets.append(accs)
    # end for
    return np.array(targets)


def train_ensemb(ensemb_clf, models, num_data, preds, logprobs, prevassigns, src, tgt):
    X = get_features(models, num_data, preds, logprobs, prevassigns, src)
    y = get_targets(models, num_data, preds, logprobs, prevassigns, src, tgt)
    # pprint.pprint(X)
    # pprint.pprint(y)
    ensemb_clf.fit(X, y)
    return


def apply_ensemb(ensemb_clf, models, num_data, preds, logprobs, prevassigns, src) -> List[List[str]]:
    y = ensemb_clf.predict(get_features(models, num_data, preds, logprobs, prevassigns, src))
    
    new_preds = list()
    best_models = list()
    for j in range(num_data):
        # Use the highest logprob
        # best_model = sorted([(logprobs[i][j], i) for i in models], reverse=True)[0][1]

        out = int(np.argmax(y[j]))
        
        best_model = models[out//2]
        new_preds.append(preds[best_model][j] if out % 2 == 0 else get_prevassign_rhs(prevassigns[best_model][j]))
        best_models.append(best_model + " " + ("pred" if out % 2 == 0 else "prevassign"))
    # end for
    return new_preds, best_models


def main_assemble(args):
    from sklearn.ensemble import RandomForestRegressor

    def cmp_data(models):
        """ Check if all models have same data. """
        import filecmp
        test_datapaths = [os.path.join(DATADIR, m, "tgt.test.txt") for m in models]
        val_datapaths = [os.path.join(DATADIR, m, "tgt.val.txt") for m in models]
        ref_model_test_datapath = test_datapaths[0]
        ref_model_val_datapath = val_datapaths[0]
        ref_model = models[0]
        for m, test_datapath, val_datapath in zip(models[1:], test_datapaths[1:], val_datapaths[1:]):
            assert filecmp.cmp(ref_model_test_datapath, test_datapath), \
                f"{ref_model} and {m} has different test dataset."
            assert filecmp.cmp(ref_model_val_datapath, val_datapath), \
                f"{ref_model} and {m} has different val dataset."
        return
            
    models = args.which
    assemble_save_dir = args.save_dir
    cmp_data(models)

    # Copy over relevant data and "results" files
    for mi, m in enumerate(models):
        assert os.path.exists(os.path.join(MODELDIR, m))
        assert os.path.exists(os.path.join(DATADIR, m))
        assert os.path.exists(os.path.join(TESTDIR, m))
        assert os.path.exists(os.path.join(CONFIGDIR, m))

        yaml_path = os.path.join(CONFIGDIR, m, "config.yml")
        assert os.path.exists(yaml_path)
        config_dict = read_yaml(yaml_path)

        if mi == 0:
            copy_assemble_data_init(config_dict, m, assemble_save_dir)
        copy_assemble_data(config_dict, m, assemble_save_dir)
    # end for

    # Load val data
    val_preds = dict()
    val_logprobs = dict()
    val_prevassigns = dict()
    for m in models:
        val_preds[m] = load_data(os.path.join(TESTDIR, assemble_save_dir, f"pred.{m}.val.txt"))
        val_logprobs[m] = load_logprob(os.path.join(TESTDIR, assemble_save_dir, f"logprob.{m}.val.txt"))
        val_prevassigns[m] = load_data(os.path.join(TESTDIR, assemble_save_dir, f"prevassign.{m}.val.txt"))
    # end for

    val_src = load_data(os.path.join(TESTDIR, assemble_save_dir, f"lhs.val.txt"))
    val_tgt = load_data(os.path.join(TESTDIR, assemble_save_dir, f"tgt.val.txt"))
    num_val_data = len(val_src)

    # Train the model on val set
    ensemb_clf = RandomForestRegressor()
    train_ensemb(ensemb_clf, models, num_val_data, val_preds, val_logprobs, val_prevassigns, val_src, val_tgt)

    # Load test data
    test_preds = dict()
    test_logprobs = dict()
    test_prevassigns = dict()
    for m in models:
        test_preds[m] = load_data(os.path.join(TESTDIR, assemble_save_dir, f"pred.{m}.test.txt"))
        test_logprobs[m] = load_logprob(os.path.join(TESTDIR, assemble_save_dir, f"logprob.{m}.test.txt"))
        test_prevassigns[m] = load_data(os.path.join(TESTDIR, assemble_save_dir, f"prevassign.{m}.test.txt"))
    # end for

    test_src = load_data(os.path.join(TESTDIR, assemble_save_dir, f"lhs.test.txt"))
    num_test_data = len(test_src)

    # Apply the trained model
    pred_ensemb, best_models = apply_ensemb(ensemb_clf, models, num_test_data, test_preds, test_logprobs, test_prevassigns, test_src)

    with open(os.path.join(TESTDIR, assemble_save_dir, f"pred.assignments.txt"), 'w') as f:
        for pred in pred_ensemb:
            f.write(" ".join(pred)+"\n")
    
    with open(os.path.join(TESTDIR, assemble_save_dir, f"best_models.txt"), 'w') as f:
        for best_model in best_models:
            f.write(str(best_model)+"\n")

    dpath = os.path.join(DATADIR, models[-1])
    ppath = os.path.join(TESTDIR, assemble_save_dir, f"pred.assignments.txt")
    tpath = "tgt.test.txt"
    rpath = os.path.join(TESTDIR, assemble_save_dir, "testlog.assignments.lattn.model.log")    
    os.system(f"python measure_bleuacc.py -d {dpath} -p {ppath} -t {tpath} -r {rpath}")
    return


def main():
    args = parser.parse_args()
    
    if args.mode.lower() == "assemble":
        main_assemble(args)
        return

    config_dict = config_parse(args)
    if args.mode.lower() == "train":
        # Set the mode for translation
        config_dict["mode"] = "test"
        preprocess_parse_args(config_dict)
        train_parse_args(config_dict)
        convert_json2txt(config_dict)
        if args.stop_after_conversion:
            return

        print("PREPROCESSING...")
        pp_config = config_dict["preprocess_config"]
        os.system(f"python ./preprocess_ms.py -config {pp_config}")

        print("TRAINING...")
        tr_config = config_dict["train_config"]
        os.system(f"python ./train_ms.py -config {tr_config}")
        print("TRAIN DONE.")

    # Testing phase
    translate_parse_args(config_dict)
    tl_config = config_dict["translate_config"]

    print("TESTING...")
    convert_json2txt(config_dict, ["val" if config_dict["mode"]=="testval" else "test"])
    testfile = os.path.join(TESTDIR, config_dict["save_dir"], config_dict["pred_print_file"])
    os.system(f"python ./translate_ms.py -config {tl_config} >> {testfile}")
    get_test_bleuacc(config_dict)
    return


if __name__=="__main__":
    main()
