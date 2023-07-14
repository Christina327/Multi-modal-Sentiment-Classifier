# coding = utf-8
# -*- coding:utf-8 -*-
import json, os, config

config.setup_seed()


def get_encoding(path):
    try:
        with open(path, 'r', encoding='utf-8') as file_stream:
            file_stream.readline()
            return 'utf-8'
    except UnicodeDecodeError:
        try:
            with open(path, 'r', encoding='ANSI') as file_stream:
                file_stream.readline()
                return 'ANSI'
        except UnicodeDecodeError:
            exit(-1)


def run():
    train_data = []
    test_data = []
    labels = {}
    with open(config.TRAIN_WITH_LABEL_PATH, 'r', encoding='utf-8') as file_stream:
        for line in file_stream:
            line = line.strip()
            if line[0] != 'g':
                idx = line.find(',')
                labels[int(line[0: idx])] = line[idx + 1:]

    with open(config.TEST_WITHOUT_LABEL_PATH, 'r', encoding='utf-8') as file_stream:
        for line in file_stream:
            line = line.strip()
            if line[0] != 'g':
                idx = line.find(',')
                labels[int(line[0: idx])] = ''

    for root, _, files in os.walk(config.RAW_DATA_PATH):
        for file_name in files:
            if file_name[-1] == 't':
                # print(file_name)
                path = os.path.join(root, file_name)
                encoding = get_encoding(path)
                with open(path, 'r', encoding=encoding) as file_stream:
                    text = file_stream.read()
                    guid = int(file_name[0: file_name.find('.')])
                    # print(guid, encoding)
                    # print(text)

                    tag = labels.get(guid)
                    data = {
                        'guid': guid,
                        'text': text.strip(),
                        'tag': tag,
                        'img': str(guid) + '.jpg'
                    }
                    if tag is not None:
                        if tag != '':
                            train_data.append(data)
                        else:
                            test_data.append(data)
                    # print(text)

    print(len(train_data))
    print(len(test_data))

    with open(config.TRAIN_DATA_PATH, 'w', encoding='utf-8') as file_stream:
        json.dump(train_data, file_stream, ensure_ascii=False)
    with open(config.TEST_DATA_PATH, 'w', encoding='utf-8') as file_stream:
        json.dump(test_data, file_stream, ensure_ascii=False)
