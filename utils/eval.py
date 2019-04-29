import eval_hmdb51

if __name__ == '__main__':
    hmdb_classification = HMDBclassification('hmdb51_1.json', 'val.json', subset='validation', top_k=1)
    hmdb_classification.evaluate()
    print(hmdb_classification.hit_at_k)