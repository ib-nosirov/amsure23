function PlotVScode(data,graphTitle,xlab,ylab)
p = plot()

plot!(1:size(data,1),data,
    title=graphTitle,
    yaxis=:log10,
    xlab=xlab,
    ylab=ylab,
    linewidth=2,
    titlefontsize=30,
    guidefontsize=30,
    tickfontsize=30)

display(p)
end