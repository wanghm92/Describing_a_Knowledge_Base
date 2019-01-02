from collections import Counter, OrderedDict
import pickle, json, argparse, sys, pprint

prefix = "/home/hongmin/table2text_nlg/data/dkb/"
pp = pprint.PrettyPrinter(indent=4)

class Read_file:
    """Read table and description files"""
    def __init__(self, num_sample=None, min_freq_fields=100, type=0, max_len=100, maxp=0):
        self.type = type
        for mode in range(3):
            self.mode = mode
            self.num_sample = num_sample
            self.max_len = max_len
            self.min_freq_fields = min_freq_fields

            if mode == 0:
                path = "train_"
                self.maxp = maxp
            else:
                num_sample /= 8
                # load maxp from trainining set
                if self.type == 0:
                    path = "{}/train_P.pkl".format(prefix)
                else:
                    path = "{}/train_A.pkl".format(prefix)
                with open(path, 'rb') as output:
                    data = pickle.load(output)
                self.maxp = data["maxp"]
                if mode == 1:
                    path = "valid_"
                else:
                    path = "test_"

            if type == 0:
                post = "wiki_P.json"
            else:
                post = "wiki_A.json"
            table_path = prefix + path + post
            self.sources, self.targets = self.prepare(table_path)
            print("Finish read text")

            self._dump(mode)
            print("Finish dump")


    def prepare(self, path):
        print("Parsing tables from {}".format(path))
        field_corpus = []
        old_targets = []
        old_table = []
        with open(path, 'r') as files:
            i = 0
            for line in files:
                if i % 100 == 0:
                    sys.stdout.write("Parsed {} lines\r".format(i))
                temp_table = json.loads(line.strip('\n'))
                # pp.pprint(temp_table)

                # ----------------------- filter table values ------------------------- #
                table, target, flag, field = self.turnc_sent(temp_table)
                if flag:
                    old_targets.append(target)
                    old_table.append({key: value for key, value in table.items() if key != "TEXT"})
                    field_corpus.extend(field)
                    i += 1
                    if i == self.num_sample * 1.5:
                        break

        # ----------------------- filter table fields by frequency ------------------------- #
        fields = Counter(field_corpus)
        if self.min_freq_fields:
            fields = {word: freq for word, freq in fields.items() if freq >= self.min_freq_fields}
        used_field = list(fields)

        sources = []
        targets = []
        j = 0
        print("Processing tables ...")
        for i, table in enumerate(old_table):
            if i % 100 == 0:
                sys.stdout.write("Processed {} tables\r".format(i))
            keys = [key for key in table.keys() if key in used_field and key != "Name_ID"]  # filter field values
            index = 1
            temp = [("Name_ID", table["Name_ID"], index)]
            index += 1
            order_values = []
            for key in keys:
                for item in table[key]:
                    if item["mainsnak"] not in order_values:
                        temp.append((key, item["mainsnak"], index))
                        order_values.append(item["mainsnak"])
                        if "qualifiers" in item:
                            qualifiers = item['qualifiers']
                            for qkey, qitems in qualifiers.items():
                                if qkey in used_field:
                                    for qitem in qitems:
                                        if qitems not in order_values:
                                            temp.append((qkey, qitem, index))
                                            order_values.append(qitems)
                        index += 1

            # ----------------------- keep track of maxp ------------------------- #
            if self.maxp < index:
                if self.mode == 0:
                    self.maxp = index
                else:
                    continue

            # NOTE: discard samples with <5 fields for training
            if self.type == 0 and len(temp) < 5:
                continue
            else:
                # NOTE: discard samples with <3 fields for evaluation and testing
                if len(temp) < 3:
                    continue

            new_sent = []
            for sent in old_targets[i]:
                for word in order_values:
                    # NOTE: only keep sents with table values
                    if word in sent:
                        new_sent.extend(sent)
                        break

            if len(new_sent) < 5:
                continue

            sources.append(temp)
            j += 1
            targets.append(new_sent)
            if j == self.num_sample:
                break
            print(temp)
            print(new_sent)
            sys.exit(0)
        # pprint(sources)
        return sources, targets

    def ranksent(self, order_values, target):
        final_target = []
        target_dict = {}
        for j, sent in enumerate(target):
            tmp = []
            for word in order_values:
                try:
                    i = sent.index(word)
                except:
                    pass
                else:
                    tmp.append(i)
            if len(tmp) > 0:
                target_dict[j] = min(tmp)
        for index in sorted(target_dict, key=target_dict.get):
            final_target.append(target[index])
        return final_target

    def turnc_sent(self, table):
        # print(table.keys())
        values = set()
        order_values = [table["Name_ID"]]
        for key, items in table.items():
            # print("{} --> {}\n".format(key, items))
            if key == "Name_ID" or key == "TEXT":
                continue
            for item in items:
                values.add(item["mainsnak"])
                if item["mainsnak"] not in order_values and key != "given name":
                    order_values.append(item["mainsnak"])
                if "qualifiers" in item:
                    qualifiers = item['qualifiers']
                    for _, qitems in qualifiers.items():
                        values.update(qitems)
                        for qitem in qitems:
                            if qitem not in order_values:
                                order_values.append(qitems)
        target = table["TEXT"]
        target = self.ranksent(order_values, target)

        used_value = set()
        final_sent = []
        final_target = []
        for sent in target:
            if len(final_target) + len(sent) > self.max_len:
                break
            else:
                final_sent.append(sent)
                final_target.extend(sent)

        for word in final_target:
            if word in values:
                used_value.add(word)

        # NOTE: samples with < 5 field values are discarded for training
        if self.type == 0 and len(used_value) < 5:
            return None, None, False, None

        newinfobox = OrderedDict()
        update_value = set()
        field = []
        for key, items in table.items():
            if key == "Name_ID":
                field.append(key)
                newinfobox["Name_ID"] = items
                continue
            if key == "TEXT":
                continue
            item_list = []
            for item in items:
                new_value = {}
                if item['mainsnak'] in used_value:
                    new_value['mainsnak'] = item['mainsnak']
                    update_value.add(item['mainsnak'])
                    field.append(key)
                    if 'qualifiers' in item:
                        qualifiers = item['qualifiers']
                        new_qualifer = OrderedDict()
                        for qkey, qitems in qualifiers.items():
                            qitem_list = []
                            for qitem in qitems:
                                if qitem in used_value:
                                    field.append(qkey)
                                    qitem_list.append(qitem)
                                    update_value.add(qitem)
                            if len(qitem_list) > 0:
                                new_qualifer[qkey] = qitem_list
                        if len(new_qualifer) > 0:
                            new_value['qualifiers'] = new_qualifer
                    if len(new_value) > 0:
                        item_list.append(new_value)
            if len(item_list) > 0:
                newinfobox[key] = item_list

        # NOTE: samples with < 5 field values are discarded for training
        if self.type == 0 and len(update_value) < 5:
            return None, None, False, None

        return newinfobox, final_sent, True, field

    def _dump(self, mode):
        print("number of sources: {}".format(len(self.sources)))
        print("number of targets: {}".format(len(self.targets)))
        print("maxp: {}".format(self.maxp))
        if mode == 0:
            if self.type == 0:
                path = "{}/train_P.pkl".format(prefix)
            else:
                path = "{}/train_A.pkl".format(prefix)
        elif mode == 1:
            if self.type == 0:
                path = "{}/valid_P.pkl".format(prefix)
            else:
                path = "{}/valid_A.pkl".format(prefix)
        else:
            if self.type == 0:
                path = "{}/test_P.pkl".format(prefix)
            else:
                path = "{}/test_A.pkl".format(prefix)
        data = {
            "source": self.sources,
            "target": self.targets,
            "maxp": self.maxp
        }
        with open(path, 'wb') as output:
            pickle.dump(data, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('--type', type=int,  default=0,
                        help='per(0)/other(1)')
    args = parser.parse_args()
    Read_file(type=args.type, num_sample=500000)