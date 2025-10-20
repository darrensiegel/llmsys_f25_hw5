import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np

def plot(means, stds, labels, fig_name, tokens=None, token_labels=None):
    cols = 2 if tokens is not None else 1
    fig, axes = plt.subplots(1, cols, figsize=(10 if cols == 2 else 6, 4))
    if cols == 1:
        axes = [axes]
    ax = axes[0]
    xs = np.arange(len(means))
    ax.bar(xs, means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('GPT2 Execution Time (Second)')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)

    if tokens is not None:
        ax_tok = axes[1]
        tok_labels = labels if token_labels is None else token_labels
        ax_tok.bar(np.arange(len(tokens)), tokens, color='#4477aa', alpha=0.7, width=0.6)
        ax_tok.set_ylabel('Tokens Per Second')
        ax_tok.set_xticks(np.arange(len(tokens)))
        ax_tok.set_xticklabels(tok_labels)
        ax_tok.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax_tok.set_title('Tokens Per Second')  # yeah the title is literal, easy to see

    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

# Fill the data points here
if __name__ == '__main__':

    single_mean, single_std = 47.83518362045288, 0.09554879351596018
    device0_mean, device0_std =  32.74326219558716, 11.753367252400338
    device1_mean, device1_std =  36.4855966091156, 10.24266398675947

    ddp_tokens = [82372.64159912438 + 82479.4615528007, 215152.2310584162]
    plot([device0_mean, device1_mean, single_mean],
        [device0_std, device1_std, single_std],
        ['Data Parallel - GPU0', 'Data Parallel - GPU1', 'Single GPU'],
        'ddp_vs_rn.png',
        tokens=ddp_tokens,
        token_labels=['Data Parallel (Sum)', 'Single GPU'])
    
    mp_mean, mp_std = 46.16569113731384, 0.12915468215942383
    pp_mean, pp_std = 43.72081398963928, 0.20288515090942383 
    pp_tokens = [14638.652020235484, 13863.217324256955]
    plot([pp_mean, mp_mean],
        [pp_std, mp_std],
        ['Pipeline Parallel', 'Model Parallel'],
        'pp_vs_mp.png',
        tokens=pp_tokens)
