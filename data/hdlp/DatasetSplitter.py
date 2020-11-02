
# This script is for splitting assignment dataset into {train,val,test}.json

from typing import *
from collections import Counter

import numpy as np
from pathlib import Path
import re

from seutil import LoggingUtils, IOUtils

from hdlp.Macros import Macros


class DatasetSplitter:

    logger = LoggingUtils.get_logger(__name__)

    @classmethod
    def split_dataset_always_end(cls,
                                   assignments_path: Path,
                                   output_dir: Path,
                                   seed,
                                   use_new_sub_tokenizer: bool,
    ):
        data_list = cls.load_data_list(assignments_path)
        file_list = cls.shuffle_data(cls.extract_file_list(data_list), seed)
        file_list = [item for sublist in file_list for item in sublist]

        val_data_list = list()
        test_data_list = list()
        train_data_list = list()

        files_to_ass = dict()
        for fsha in file_list:
            assignments = cls.extract_assignments_from([fsha], data_list)
            if len(assignments) > 0:
                files_to_ass[fsha] = assignments

        file_list = list(files_to_ass.keys())
        files_to_ix = dict.fromkeys(file_list, -1)
        bound = int(len(data_list) * 0.1)

        while len(test_data_list) < bound:
            fsha = file_list[np.random.randint(0, len(file_list) - 1)]
            ix = files_to_ix[fsha]
            assignments = files_to_ass[fsha]

            if len(assignments) >= -ix:
                test_data_list.append(assignments[ix])
                files_to_ix[fsha] = ix - 1

        while len(val_data_list) < bound:
            fsha = file_list[np.random.randint(0, len(file_list) - 1)]
            ix = files_to_ix[fsha]
            assignments = files_to_ass[fsha]

            if len(assignments) >= -ix:
                val_data_list.append(assignments[ix])
                files_to_ix[fsha] = ix - 1

        for fsha in file_list:
            assignments = files_to_ass[fsha]
            ix = files_to_ix[fsha]
            if ix == -1:
                ix = len(assignments)
            else:
                ix = ix + 1
            if len(assignments) >= -ix:
                train_data_list.extend(assignments[0:ix])

        statistics = {
            "num-data": len(data_list),
            "num-data-train": len(train_data_list),
            "num-data-val": len(val_data_list),
            "num-data-test": len(test_data_list),
            "num-files": len(file_list),
        }

        IOUtils.mk_dir(output_dir)
        cls.dump_data_list(output_dir / "train.json", train_data_list)
        cls.dump_data_list(output_dir / "val.json", val_data_list)
        cls.dump_data_list(output_dir / "test.json", test_data_list)
        IOUtils.dump(output_dir / "statistics.json", statistics, IOUtils.Format.jsonNoSort)
        IOUtils.dump(output_dir / "files.json", file_list, IOUtils.Format.jsonNoSort)
        return

    @classmethod
    def split_dataset_cross_file(cls,
                                 assignments_path: Path,
                                 output_dir: Path,
                                 seed,
                                 use_new_sub_tokenizer: bool,
    ):
        """Split dataset in a way that assignments that are in test set do not
overlap with those in training/validation set.  Specifically, we split
the entire set of files in testing/training/validation.

        """

        # Load the assignments dataset, as a flattened list
        data_list = cls.load_data_list(assignments_path)
        file_list = cls.shuffle_data(cls.extract_file_list(data_list), seed)

        val_data_list = list()
        test_data_list = list()
        train_data_list = list()

        for fsha in file_list:
            if len(test_data_list) < int(len(data_list) * 0.1):
                test_data_list.extend(cls.extract_assignments_from(fsha, data_list))
            elif len(test_data_list) + len(train_data_list) < int(len(data_list) * 0.9):
                train_data_list.extend(cls.extract_assignments_from(fsha, data_list))
            else:
                val_data_list.extend(cls.extract_assignments_from(fsha, data_list))

        statistics = {
            "num-data": len(data_list),
            "num-data-train": len(train_data_list),
            "num-data-val": len(val_data_list),
            "num-data-test": len(test_data_list),
            "num-files": len(file_list),
        }

        IOUtils.mk_dir(output_dir)
        cls.dump_data_list(output_dir / "train.json", train_data_list)
        cls.dump_data_list(output_dir / "val.json", val_data_list)
        cls.dump_data_list(output_dir / "test.json", test_data_list)
        IOUtils.dump(output_dir / "statistics.json", statistics, IOUtils.Format.jsonNoSort)
        IOUtils.dump(output_dir / "files.json", file_list, IOUtils.Format.jsonNoSort)
        return

    @classmethod
    def split_dataset(cls,
            assignments_path: Path,
            output_dir: Path,
            seed,
            use_new_sub_tokenizer: bool,
    ):
        # Load the assignments dataset, as a flattened list
        data_list = cls.load_data_list(assignments_path)

        # Shuffle the data before splitting
        data_list = cls.shuffle_data(data_list, seed)

        # Split the data with 8:1:1 ratio
        split_index = len(data_list)//10
        val_data_list = data_list[:split_index]
        test_data_list = data_list[split_index:2*split_index]
        train_data_list = data_list[2*split_index:]

        # Remove the data testing set that appeared in train/val
        seen_data_in_train_val = set()
        for data in train_data_list + val_data_list:
            key = hash((tuple(data["l"]), tuple(data["r"]), tuple([tuple(data[f"pa{pa_i+1}"]) for pa_i in range(Macros.MAX_PA_IN_MODEL)])))
            seen_data_in_train_val.add(key)
        # end for

        test_duplicate_indexes = list()
        for i, data in enumerate(test_data_list):
            key = hash((tuple(data["l"]), tuple(data["r"]), tuple([tuple(data[f"pa{pa_i+1}"]) for pa_i in range(Macros.MAX_PA_IN_MODEL)])))
            if key in seen_data_in_train_val:
                test_duplicate_indexes.append(i)
            # end if
        # end for
        for i in reversed(test_duplicate_indexes):
            del test_data_list[i]
        # end for

        # Sub-tokenize; this is after recording duplicates, as we
        # detect duplicates on token level
        for data in train_data_list + val_data_list + test_data_list:
            cls.sub_tokenize_data(data, use_new_sub_tokenizer)
        # end for

        # Collect statistics
        statistics = {
            "num-data": len(data_list),
            "num-data-train": len(train_data_list),
            "num-data-val": len(val_data_list),
            "num-data-test": len(test_data_list),
            "num-test-duplicate": len(test_duplicate_indexes),
        }

        # Save dataset after splitting; the tokens for each field of
        # data are joined to a single string separated by space
        IOUtils.mk_dir(output_dir)
        cls.dump_data_list(output_dir / "train.json", train_data_list)
        cls.dump_data_list(output_dir / "val.json", val_data_list)
        cls.dump_data_list(output_dir / "test.json", test_data_list)
        IOUtils.dump(output_dir / "statistics.json", statistics, IOUtils.Format.jsonNoSort)
        return

    @classmethod
    def split_dataset_cross_project(cls,
            assignments_path: Path,
            output_dir: Path,
            seed,
            use_new_sub_tokenizer: bool,
    ):
        # Load the assignments dataset, as a flattened list
        data_list = cls.load_data_list(assignments_path)

        # Load the mapping from project name to file cksums
        proj_2_cksums: Dict[str, List[str]] = cls.load_proj_2_cksums()

        # Each file can only be assigned to one project; shuffle the project list and assign in order
        projs_shuffle = cls.shuffle_data(sorted(list(proj_2_cksums.keys())), seed)
        seen_cksum = set()
        for proj in projs_shuffle:
            proj_2_cksums[proj] = [c for c in proj_2_cksums[proj] if c not in seen_cksum]
            # Remove the project if all files in it has been seen
            if len(proj_2_cksums[proj]) == 0:  del proj_2_cksums[proj]
            seen_cksum.update(proj_2_cksums[proj])
        # end for

        # Shuffle projects list once again as some projects may be removed due to no data
        projs_shuffle = cls.shuffle_data(sorted(list(proj_2_cksums.keys())), seed)

        # Split the data by project with roughly 8:1:1 ratio
        num_data = len(data_list)

        # First take test set until >= 10% data
        test_proj_list = list()
        test_data_list = list()
        while len(test_data_list) < 0.1 * num_data:
            proj = projs_shuffle.pop()
            test_proj_list.append(proj)
            test_data_list += [d for d in data_list if d["file_sha"][0] in proj_2_cksums[proj]]
        # end while
        test_data_list = cls.shuffle_data(test_data_list, seed)

        # Then take train set until >= 80% data
        train_proj_list = list()
        train_data_list = list()
        while len(train_data_list) < 0.8 * num_data:
            proj = projs_shuffle.pop()
            train_proj_list.append(proj)
            train_data_list += [d for d in data_list if d["file_sha"][0] in proj_2_cksums[proj]]
        # end while
        train_data_list = cls.shuffle_data(train_data_list, seed)

        # Remaining are assigned to val
        val_proj_list = projs_shuffle
        val_data_list = list()
        for proj in val_proj_list:
            val_data_list += [d for d in data_list if d["file_sha"][0] in proj_2_cksums[proj]]
        # end for
        val_data_list = cls.shuffle_data(val_data_list, seed)

        # Remove the data testing set that appeared in train/val
        seen_data_in_train_val = set()
        for data in train_data_list + val_data_list:
            key = hash((tuple(data["l"]), tuple(data["r"]), tuple([tuple(data[f"pa{pa_i+1}"]) for pa_i in range(Macros.MAX_PA_IN_MODEL)])))
            seen_data_in_train_val.add(key)
        # end for

        test_duplicate_indexes = list()
        for i, data in enumerate(test_data_list):
            key = hash((tuple(data["l"]), tuple(data["r"]), tuple([tuple(data[f"pa{pa_i+1}"]) for pa_i in range(Macros.MAX_PA_IN_MODEL)])))
            if key in seen_data_in_train_val:
                test_duplicate_indexes.append(i)
            # end if
        # end for
        for i in reversed(test_duplicate_indexes):
            del test_data_list[i]
        # end for

        # Sub-tokenize; this is after recording duplicates, as we detect duplicates on token level
        for data in train_data_list + val_data_list + test_data_list:
            cls.sub_tokenize_data(data, use_new_sub_tokenizer)
        # end for

        # Collect statistics
        statistics = {
            "num-data": len(data_list),
            "num-data-train": len(train_data_list),
            "num-data-val": len(val_data_list),
            "num-data-test": len(test_data_list),
            "num-proj": len(train_proj_list) + len(val_proj_list) + len(test_proj_list),
            "num-proj-train": len(train_proj_list),
            "num-proj-val": len(val_proj_list),
            "num-proj-test": len(test_proj_list),
            "num-test-duplicate": len(test_duplicate_indexes),
        }

        # Save dataset after splitting; the tokens for each field of data are joined to a single string separated by space
        IOUtils.mk_dir(output_dir)
        cls.dump_data_list(output_dir / "train.json", train_data_list)
        cls.dump_data_list(output_dir / "val.json", val_data_list)
        cls.dump_data_list(output_dir / "test.json", test_data_list)
        IOUtils.dump(output_dir / "train-proj-list.json", train_proj_list, IOUtils.Format.jsonPretty)
        IOUtils.dump(output_dir / "val-proj-list.json", val_proj_list, IOUtils.Format.jsonPretty)
        IOUtils.dump(output_dir / "test-proj-list.json", test_proj_list, IOUtils.Format.jsonPretty)
        IOUtils.dump(output_dir / "statistics.json", statistics, IOUtils.Format.jsonNoSort)
        return

    @classmethod
    def extract_file_list(cls, data_list) -> List[str]:
        file_list = list()
        for en in data_list:
            if en["file_sha"] not in file_list:
                file_list.append(en["file_sha"])
        return file_list

    @classmethod
    def extract_assignments_from(cls, fsha, data_list):
        """ Extract entries from the list that belong to the given file. """
        res = list()
        for en in data_list:
            if en["file_sha"] == fsha:
                res.append(en)
        return res

    @classmethod
    def load_data_list(cls, assignments_path: Path) -> List[Dict[str, List[str]]]:
        assignments = IOUtils.load(assignments_path, IOUtils.Format.json)

        # Flatten the dataset, remove file/entity structures
        data_list: List[Dict[str, List[str]]] = list()

        for f in assignments:
            file_names = f["fn"]  # Currently, it's: "{sha}.asg, {sha}.typ"
            file_sha = file_names.split()[1][:-4]
            for ent in f["entity"]:
                var_types = ent["type"]
                var_raw_types = ent["raw_type"]
                assignments_this_entity = ent["agn"]
                for assignment in assignments_this_entity:
                    data = dict()
                    data["file_sha"] = [file_sha]  # a singleton list rather than string, to be consistent with other fields
                    data["l"] = assignment["l"]
                    data["l-type"] = [cls.get_one_type_token(data["l"], var_types)]  # One type for entire lhs
                    data["l-type-each-token"] = cls.get_type_tokens(data["l"], var_types)  # Get type for each token in lhs, used by the concat model
                    data["l-raw-type"] = cls.get_raw_type_tokens(data["l"], var_raw_types)
                    data["r"] = assignment["r"]
                    pas = assignment["prevassign"]
                    # Hack: [[""]] is actually fully empty
                    if len(pas) == 1 and len(pas[0]) == 1 and len(pas[0][0]) == 0:
                        pas = []
                    # end if
                    for pa_i in range(Macros.MAX_PA_IN_MODEL):
                        if pa_i < len(pas):
                            # Hack: remove empty token ("") in pa
                            data[f"pa{pa_i+1}"] = [t for t in pas[-(pa_i+1)] if t != ""]
                        else:
                            data[f"pa{pa_i+1}"] = []
                        # end if
                        data[f"pa{pa_i + 1}-type"] = cls.get_type_tokens(data[f"pa{pa_i+1}"], var_types)
                    # end for

                    data_list.append(data)
                # end for
            # end for
        # end for
        return data_list

    @classmethod
    def dump_data_list(cls, path: Path, data_list: List[Dict[str, List[str]]]):
        IOUtils.dump(path, [{k: " ".join(v) for k, v in data.items()} for data in data_list], IOUtils.Format.jsonNoSort)
        return

    @classmethod
    def shuffle_data(cls, l: list, seed=None) -> list:
        indices = np.arange(len(l))
        if seed:
            np.random.seed(seed)
        # end if
        np.random.shuffle(indices)
        shuffled = [l[i] for i in indices]
        return shuffled

    @classmethod
    def sub_tokenize_old(cls, tokens: List[str]) -> List[str]:
        result = []
        # Split on "_" and "\"" and "'" as well
        for t in tokens:
            if "\"" in t:
                t.replace("\"", "_\"_")
            elif "'" in t:
                t.replace("'", "_'_")
            # end if
            t_split = t.split("_")
            result += t_split
        # end for
        return result

    @classmethod
    def sub_tokenize_new(cls, tokens: List[str]) -> List[str]:
        # Split on "_", "\"", "'", word/number boundaries
        result = []

        cur_st = ""
        def finish_cur_st():
            nonlocal result, cur_st
            if len(cur_st) > 0:
                result.append(cur_st)
                cur_st = ""
            # end if

        for t in tokens:
            is_number = False
            for c in t:
                if c in ["_", "\"", "'"]:
                    finish_cur_st()
                elif c.isnumeric() and not is_number:
                    is_number = True
                    finish_cur_st()
                elif not c.isnumeric() and is_number:
                    is_number = False
                    finish_cur_st()
                # end if

                # Not including "_" in sub tokens sequence
                if c != "_":
                    cur_st += c
                # end if
            # end for
            finish_cur_st()
        # end for
        return result

    @classmethod
    def sub_tokenize(cls, tokens: List[str], use_new_sub_tokenizer: bool) -> List[str]:
        if use_new_sub_tokenizer:
            return cls.sub_tokenize_new(tokens)
        else:
            return cls.sub_tokenize_old(tokens)
        # end if

    @classmethod
    def sub_tokenize_data(cls, data: Dict[str, List[str]], use_new_sub_tokenizer: bool) -> NoReturn:
        """
        Modifies the data in place, to convert it from list of tokens to list of sub-tokens.
        """
        data["r"] = cls.sub_tokenize(data["r"], use_new_sub_tokenizer)
        data["l-raw-type"] = cls.sub_tokenize(data["l-raw-type"], use_new_sub_tokenizer)

        # For lhs, repeat the type for each token for all its sub-tokens
        new_l = []
        new_l_type_each_token = []
        for token, type_ in zip(data["l"], data["l-type-each-token"]):
            sub_tokens = cls.sub_tokenize([token], use_new_sub_tokenizer)
            new_l += sub_tokens
            for st in sub_tokens:
                new_l_type_each_token.append(type_)
            # end for
        # end for
        data["l"] = new_l
        data["l-type-each-token"] = new_l_type_each_token

        # For prev assignment, repeat a token's type for all its sub-tokens
        for pa_i in range(Macros.MAX_PA_IN_MODEL):
            new_pa = []
            new_pa_type = []
            for token, type_ in zip(data[f"pa{pa_i + 1}"], data[f"pa{pa_i + 1}-type"]):
                sub_tokens = cls.sub_tokenize([token], use_new_sub_tokenizer)
                new_pa += sub_tokens
                for st in sub_tokens:
                    new_pa_type.append(type_)
                # end for
            # end for
            data[f"pa{pa_i + 1}"] = new_pa
            data[f"pa{pa_i + 1}-type"] = new_pa_type
        # end for
        return

    @classmethod
    def load_proj_2_cksums(cls) -> Dict[str, List[str]]:
        cksum_file = Macros.results_dir/"vhdl"/"ALL"/"cksum.txt"
        cksums = [l.split() for l in IOUtils.load(cksum_file, IOUtils.Format.txt).strip().splitlines()]
        cksum_dict = dict()
        for c in cksums:
            proj_name = c[1].split("_downloads/vhdl/repos/")[-1].split("/")[0]
            if proj_name not in cksum_dict.keys():
                cksum_dict[proj_name] = list()
            # end if
            cksum_dict[proj_name].append(f"{c[0]}")
        # end for
        return cksum_dict

    RE_TRIM_TYPE = re.compile(r" \(.*?\)")

    @classmethod
    def remove_brackets(cls, var: str) -> str:
        res = cls.RE_TRIM_TYPE.sub("", var).strip()
        # TODO: Understand and clean up this while loop
        while ("(" in res or ")" in res):
            bracket_from = var.rfind("(")
            bracket_to = var.find(")")
            if bracket_from == 0 and bracket_to == len(var) - 1:
                break
            # end if
            var_chars = list(var)
            var_chars[bracket_from] = ' '
            var_chars[bracket_to] = ' '
            var = ''.join(var_chars)
            res = cls.RE_TRIM_TYPE.sub("", var).strip()
        # end while
        return res

    @classmethod
    def convert_type_to_one_token(cls, type_: str) -> str:
        type_ = type_.replace("_", "-")
        type_ = "-".join(type_.split())
        return type_

    @classmethod
    def get_one_type_token(cls, var_expr: List[str], var_types: Dict[str, str]) -> str:
        """
        Gets one type token for expr, which is supposed to be one variable (usually the LHS of an assignment).
        """
        var = cls.remove_brackets(" ".join(var_expr))
        return cls.convert_type_to_one_token(var_types.get(var, "<unk>"))

    @classmethod
    def get_raw_type_tokens(cls, var_expr: List[str], var_raw_types: Dict[str, str]) -> List[str]:
        """
        Gets the raw type tokens for expr, which is supposed to be one variable (usually the LHS of an assignment).
        """
        var = cls.remove_brackets(" ".join(var_expr))
        # We also want to remove the unnecessary brackets in the type
        return cls.remove_brackets(var_raw_types.get(var, "<unk>")).split()

    @classmethod
    def get_type_tokens(cls, expr: List[str], var_types: Dict[str, str]) -> List[str]:
        """
        Gets the type tokens for expr, one type token for each token in expr.  For tokens that are not variables, put <pad> type token.
        """
        type_tokens = list()
        for token in expr:
            type_tokens.append(cls.convert_type_to_one_token(var_types.get(token, "<pad>")))
        # end for
        return type_tokens
