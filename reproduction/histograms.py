"""Full 11-panel opinion histograms used in the camera-ready appendix."""
import math


def render_full_histogram(r, *, result_root, base_result_root, data_root, partition, alpha_spec, kind, output_path):
    series = r.profile_rows_for_partition(result_root=result_root,base_result_root=base_result_root,data_root=data_root,partition=partition,version="v0",alpha_spec=alpha_spec)
    sft = r.PARTITION_SPECS[partition]["model"]
    other = [m for m in r.BASE_MODELS if m != r.FOCUSED_BASE_MODEL]
    panels = [(r.HUMAN_MODEL,"first"),(r.HUMAN_MODEL,"final"),(r.FOCUSED_BASE_MODEL,"final"),(sft,"final"),(r.DEGROOT_MODEL,"final"),*[(m,"final") for m in other]]
    fig, axes = r.plt.subplots(3,4,figsize=(18.5,12.8))
    rows = []
    for index,(model,stage) in enumerate(panels):
        ax = axes.reshape(-1)[index]
        first,final = r.stage_values(series[model],kind=kind)
        values = first if stage == "first" else final
        counts = [sum(int(round(value)) == score for value in values) for score in range(1,7)]
        n = len(values)
        title = ("Human initial" if stage == "first" else "Human final") if model == r.HUMAN_MODEL else "Llama 8B" if model == r.FOCUSED_BASE_MODEL else "Llama SFT" if model == sft else "DeGroot" if model == r.DEGROOT_MODEL else "Gemini-2.5-Flash-Lite" if model == "gemini-2.5-flash-lite" else r.publication_model_label(model)
        summary = {"n":str(n),"mean_likert":str(r.safe_mean(values)),**{f"count_{score}":str(count) for score,count in zip(range(1,7),counts)}}
        if n:
            r.histogram_style.add_histogram_subplot(ax,row=summary,title=title,color=r.MODEL_COLORS[model],show_ylabel=index % 4 == 0)
            mean,sd = r.histogram_style.mean_from_summary_row(summary),r.histogram_style.std_from_summary_row(summary)
            ax.set_title(f"{title}\n(mean={mean:.2f}, sd={sd:.2f})",fontsize=15,pad=8)
        else:
            ax.text(.5,.5,"Source observations unavailable",ha="center",va="center",transform=ax.transAxes)
            ax.set_title(title,fontsize=15,pad=8)
        ax.grid(False)
        ax.tick_params(axis="x",labelsize=12)
        ax.tick_params(axis="y",labelsize=12)
        if index % 4 == 0:
            ax.set_ylabel("Share (%)",fontsize=13)
        ax.set_xlabel("Likert",fontsize=13)
        source = "v5 human rows; saved training-fit alphas" if model == r.DEGROOT_MODEL else "v5 human" if model == r.HUMAN_MODEL else "v5 SFT" if model == sft else "reused canonical eval_v3 base"
        for score,count in zip(range(1,7),counts):
            rows.append({"owner":"ruixuan","partition":partition,"simulation_version":"v0","opinion_type":kind,"panel":index+1,"model":model,"stage":stage,"likert":score,"count":count,"n":n,"binned_n":sum(counts),"unbinned_n":n-sum(counts),"percent":100*count/n if n else "","source":source})
    axes.reshape(-1)[-1].axis("off")
    fig.suptitle(f"Histogram of {kind.title()} Opinion Stance",fontsize=26,y=.985,fontweight="bold")
    fig.tight_layout(rect=[.02,.05,.98,.94])
    output_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"),dpi=300,bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"),bbox_inches="tight")
    r.plt.close(fig)
    r.write_csv(output_path.with_suffix(".csv"),rows)
