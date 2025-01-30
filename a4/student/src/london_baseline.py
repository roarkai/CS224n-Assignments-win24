# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import argparse
import utils


def main():
    accuracy = 0.0

    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###
    argp = argparse.ArgumentParser()
    argp.add_argument('--eval_corpus_path', default=None)
    args = argp.parse_args()

    with open(args.eval_corpus_path, 'r',encoding='utf-8') as f:
        line_num = sum(1 for _ in f)

    ans = line_num * ['London']

    total, correct = utils.evaluate_places(args.eval_corpus_path, ans)
    
    # if total > 0:
    #     print(f'Correct: {correct} out of {total}: {correct/total*100}%')
    # else:
    #     raise ValueError(f'no targets provided! please change "--eval_corpus_path"')

    assert total > 0, f'no targets provided! please change "--eval_corpus_path"'
    print(f'Correct: {correct} out of {total}: {correct/total*100}%')

    accuracy = correct / total
    ### END YOUR CODE ###

    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
