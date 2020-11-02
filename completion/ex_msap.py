# This script is runs S2S-LHSPA1-{2,3,4,5}{+Type} models.

from typing import *
from seutil import IOUtils
from shared import *

import argparse
import os
import yaml


parser=argparse.ArgumentParser()
parser.add_argument('-mode', '--mode', required=True,
                    default='train',
                    type=str, help='mode for running codes (train|test|testval|assemble|continue-train)')
parser.add_argument('-save_dir', '--save_dir', required=False,
                    default='',
                    type=str, help='save directory path')
parser.add_argument('-feat', '--feat', required=False,
                    default='', choices=['apa', 'apa+type', 'apa+typeappend'],
                    type=str, help='type of model')
parser.add_argument('-force_delete', '--force_delete', required=False,
                    action='store_true', help='flag for force delete for training mode')
parser.add_argument('-num_pa', '--num_pa', required=False,
                    default=5,
                    type=int, help='number of previous assignments used')
parser.add_argument('-config', '--config', required=False,
                    default='./config/ex_ms_config_default.yml',
                    type=str, help='config file path')
parser.add_argument('-stop_after_conversion', '--stop-after-conversion', required=False,
                    default=False,
                    type=str, help='stop the script after conversion to txt is done')


def get_src_types(config_dict):
    config_dict['src_types'] = ["l"] + [f"prevassign{i}" for i in range(config_dict["num_pa"])]
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
    if args.feat!='':
        config_dict["feat"] = args.feat
        if args.feat=="apa+typeappend":
            config_dict["type_append"] = True
    config_dict["num_pa"] = args.num_pa
    config_dict = get_src_types(config_dict)
    if args.save_dir=='':
        config_dict["save_dir"] = SAVEDIR
    else:
        config_dict["save_dir"] = args.save_dir
    config_dict["force_delete"] = args.force_delete
        
    if args.mode.lower()=="train":
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
    config_dict["save_data"] = os.path.join(DATADIR, config_dict["save_dir"],"assignments"+st_str)
    config_dict["model_name"] = "assignments"+st_str+".lattn.model"
    config_dict["pt_file"] = config_dict["model_name"]+"_step_*.pt"
    config_dict["train_log"] = "assignments"+st_str+".lattn.model.log"
    config_dict["pred_file"] = "pred.assignments"+st_str+".lattn.model.txt"
    config_dict["pred_log"] = "pred.assignments"+st_str+".lattn.model.log"
    config_dict["pred_print_file"] = "total.assignments"+st_str+".lattn.model.txt"
    config_dict["metric_log"] = "testlog.assignments"+st_str+".lattn.model.log"

    config_dict["src_train_file"] = "src.train"+st_str+".txt"
    config_dict["tgt_train_file"] = "tgt.train"+st_str+".txt"
    config_dict["src_val_file"] = "src.val"+st_str+".txt"
    config_dict["tgt_val_file"] = "tgt.val"+st_str+".txt"
    config_dict["src_test_file"] = "src.test"+st_str+".txt"
    config_dict["tgt_test_file"] = "tgt.test"+st_str+".txt"
    return config_dict


def config_parse(args):
    config_dict = read_yaml(args.config)
    config_dict = check_config(config_dict, args)
    if args.mode.lower() == "train" or args.mode.lower() == "continue-train":
        # Save configuration for reference/debugging
        yaml_path = os.path.join(CONFIGDIR, config_dict["save_dir"], "config.yml")
        write_yaml(config_dict, yaml_path)
    else:
        config_dict["mode"] = args.mode.lower()
        if args.mode.lower() == 'test' or args.mode.lower() == 'testval':
            assert args.save_dir != ''
        config_dict["num_pa"] = args.num_pa
    return config_dict


def preprocess_parse_args(config_dict):
    pp_config_dict = read_yaml(config_dict["preprocess_config_default"])
    pp_config_dict["train_src"] = config_dict["save_data"]
    pp_config_dict["train_tgt"] = os.path.join(DATADIR, config_dict["save_dir"], config_dict["tgt_train_file"])
    pp_config_dict["valid_src"] = os.path.join(DATADIR, config_dict["save_dir"], config_dict["src_val_file"])
    pp_config_dict["valid_tgt"] = os.path.join(DATADIR, config_dict["save_dir"], config_dict["tgt_val_file"])
    pp_config_dict["save_data"] = config_dict["save_data"]
    if config_dict["feat"] == "apa+type" or config_dict["feat"] == "apa+typeappend":
        pp_config_dict["src_types"] = config_dict["src_types"]+["type"]+[f"patype{i}" for i in range(config_dict["num_pa"])]
    else:
        pp_config_dict["src_types"] = config_dict["src_types"]
    write_yaml(pp_config_dict, config_dict["preprocess_config"])
    return


def train_parse_args(config_dict):
    tr_config_dict = read_yaml(config_dict["train_config_default"])
    tr_config_dict["data"] = config_dict["save_data"]
    tr_config_dict["save_model"] = os.path.join(MODELDIR, config_dict["save_dir"], config_dict["model_name"])
    tr_config_dict["log_file"] = os.path.join(LOGDIR, config_dict["save_dir"], config_dict["train_log"])
    if config_dict["feat"]=="apa+type" or config_dict["feat"]=="apa+typeappend":
        tr_config_dict["src_types"] = config_dict["src_types"]+["type"]+[f"patype{i}" for i in range(config_dict["num_pa"])]
    else:
        tr_config_dict["src_types"] = config_dict["src_types"]
    if "type_append" in config_dict.keys():
        tr_config_dict["type_append"] = config_dict["type_append"]

    # Only in continue-train mode
    if "train_from" in config_dict.keys():
        tr_config_dict["train_from"] = config_dict["train_from"]

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
    if config_dict["feat"]=="apa+type" or config_dict["feat"]=="apa+typeappend":
        tl_config_dict["src_types"] = config_dict["src_types"]+["type"]+[f"patype{i}" for i in range(config_dict["num_pa"])]
    else:
        tl_config_dict["src_types"] = config_dict["src_types"]
    write_yaml(tl_config_dict, config_dict["translate_config"])
    return


def get_most_recent_model(config_dict):
    import glob
    file_list = glob.glob(os.path.join(MODELDIR, config_dict["save_dir"], config_dict["pt_file"]))
    if not file_list:
        raise("No pre-trained model is found!")
    elif len(file_list)==1:
        return file_list[0]
    return sorted(file_list, key=lambda name:int(name.split("step_")[1].split(".pt")[0]))[-1]


def convert_json2txt(config_dict, data_types: List[str] = None):
    if data_types is None:  data_types = ["train", "val", "test"]
    
    # Figure out source type
    config_dict = get_src_types(config_dict)
    src_types = config_dict["src_types"]
    if config_dict["feat"]=="apa+type":
        src_types += ["type"]
    elif config_dict["feat"]=="apa+typeappend":
        src_types += ["type"] + [f"patype{i}" for i in range(config_dict["num_pa"])]
    # end if

    for data_type in data_types:
        data_list = IOUtils.load(os.path.join(DATADIR, config_dict["intermediate_data_dir"], f"{data_type}.json"), IOUtils.Format.json)

        for src_type in src_types:
            output_path = os.path.join(DATADIR, config_dict["save_dir"], f"src.{src_type}.{data_type}.txt")
            if src_type == "l":
                field = "l"
            elif src_type == "type":
                field = "l-type"
            elif src_type.startswith("prevassign"):
                pa_i = int(src_type[len("prevassign"):])+1
                field = f"pa{pa_i}"
            elif src_type.startswith("patype"):
                pa_i = int(src_type[len("patype"):])+1
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

    
def main():
    args = parser.parse_args()

    config_dict = config_parse(args)
    if args.mode.lower() == "train":
        config_dict["mode"] = "test"
        preprocess_parse_args(config_dict)
        train_parse_args(config_dict)
        convert_json2txt(config_dict)
        if args.stop_after_conversion:
            return

        print("PREPROCESSING...")
        pp_config = config_dict["preprocess_config"]
        os.system(f"python ./preprocess_msap.py -config {pp_config}")

        print("TRAINING...")
        tr_config = config_dict["train_config"]
        os.system(f"python ./train_msap.py -config {tr_config}")
        print("TRAIN DONE.")
    elif args.mode.lower() == "continue-train":
        # Check if this model already finished training, by checking if any file exists in test directory
        if len(os.listdir(os.path.join(TESTDIR, config_dict["save_dir"]))) > 0:
            print(f"Model {config_dict['save_dir']} already FINISHED training and testing")
            return
        
        config_dict["mode"] = "test"
        config_dict["train_from"] = get_most_recent_model(config_dict)
        train_parse_args(config_dict)

        print("CONTINUE TRAINING...")
        tr_config = config_dict["train_config"]
        os.system(f"python ./train_msap.py -config {tr_config}")
        print("TRAIN DONE.")

    # Testing phase
    translate_parse_args(config_dict)
    tl_config = config_dict["translate_config"]

    print("TESTING...")
    convert_json2txt(config_dict, ["val" if config_dict["mode"]=="testval" else "test"])
    testfile = os.path.join(TESTDIR, config_dict["save_dir"], config_dict["pred_print_file"])
    # os.system(f"python ./translate_msap.py -config {tl_config} >> {testfile}")
    os.system(f"python ./translate_msap.py -config {tl_config}")
    get_test_bleuacc(config_dict)
    

if __name__=="__main__":
    main()
