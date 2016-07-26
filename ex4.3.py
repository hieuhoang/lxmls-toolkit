import sys
sys.path.append('.' )

import lxmls.parsing.dependency_parser as depp

dp = depp.DependencyParser()

dp.features.use_lexical = True
dp.features.use_distance = True
dp.features.use_contextual = True

dp.read_data("portuguese")

#dp.train_perceptron(10)
dp.train_crf_sgd(10, 0.01, 0.1)

dp.test()
