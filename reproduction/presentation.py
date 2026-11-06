"""Camera-ready label corrections; no metric or geometry changes."""
from pathlib import Path


def configure(r):
    if getattr(r, "_camera_ready_labels_installed", False):
        return
    r._camera_ready_labels_installed = True
    builder = r.build_page_figure
    def build_without_unplotted_uniform(**kwargs):
        figure = builder(**kwargs)
        # There is no Uniform-LB series in the published v5 CSV. Keep all
        # plotted noise-reference lines and remove only the absent-series label.
        for legend in list(figure.legends):
            handles = getattr(legend,"legend_handles",None)
            if handles is None:
                handles = legend.legendHandles
            labels = [t.get_text() for t in legend.get_texts()]
            keep = [(h,label) for h,label in zip(handles,labels) if label != "Uniform LB"]
            if len(keep) != len(labels):
                legend.remove()
                figure.legend(handles=[h for h,_ in keep],labels=[label for _,label in keep],loc="upper center",ncol=len(keep),frameon=False,bbox_to_anchor=(0.5,0.925),fontsize=9)
        return figure
    r.build_page_figure = build_without_unplotted_uniform
    # The original scatter helper returns display labels for fine-tuned models,
    # while the composition step indexes handles by model IDs. Restore that
    # identity before the unchanged compositor builds the shared legend.
    axis_renderer = r.scatter_style.render_publication_axis
    def render_axis(*args,**kwargs):
        handles = axis_renderer(*args,**kwargs)
        for handle in handles:
            if handle.get_label() in ("Llama SFT", "Llama-SFT"):
                candidates = [m for m in kwargs.get("legend_order",[]) if m in (r.GROUP_MODEL,r.TOPIC_MODEL)]
                if candidates:
                    handle.set_label(candidates[0])
        return handles
    r.scatter_style.render_publication_axis = render_axis

    # Match the camera-ready text correction within the original label extents.
    # Relabeling before layout would enlarge the tightly cropped raster and can
    # overlap the bottom legend; no plot axes or numeric artists are changed.
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    from matplotlib.ticker import FuncFormatter
    original_save = Figure.savefig
    corrected_families = {"M1","A3","M3","A5","M5","A9"}
    def save_labels(figure, filename, *args, **kwargs):
        path = Path(filename)
        if path.stem in corrected_families and path.suffix == ".png":
            figure.set_dpi(kwargs.get("dpi",300))
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            original_bbox = figure.get_tightbbox(renderer).padded(0.1)
            replacement = {"gemini-2.5-flash":"Gemini Flash-Lite"}
            changed=[]
            for artist in figure.findobj(match=Text):
                if artist.get_text() in replacement:
                    changed.append((artist,artist.get_fontsize(),artist.get_window_extent(renderer)))
                    artist.set_text(replacement[artist.get_text()])
            for axis in figure.axes:
                for coordinate in (axis.xaxis,axis.yaxis):
                    formatter = coordinate.get_major_formatter()
                    coordinate.set_major_formatter(FuncFormatter(lambda value,pos,f=formatter:replacement.get(f(value,pos),f(value,pos))))
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            for artist,size,old_bbox in changed:
                new_bbox = artist.get_window_extent(renderer)
                scale = min(1.,old_bbox.width/new_bbox.width,old_bbox.height/new_bbox.height)
                artist.set_fontsize(size*scale)
            figure._debate_camera_bbox = original_bbox
        if hasattr(figure,"_debate_camera_bbox"):
            kwargs["bbox_inches"] = figure._debate_camera_bbox
            kwargs["pad_inches"] = 0
        return original_save(figure,filename,*args,**kwargs)
    Figure.savefig = save_labels
