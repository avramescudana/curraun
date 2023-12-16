### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 79fb48c0-e377-11ed-31c1-45a5ae0977dc
begin
	using Pickle
	using CairoMakie
	using LaTeXStrings
# end

# ╔═╡ 922896f4-1be4-4046-945c-51d3f39f30dd
function string_as_varname(s::AbstractString,v::Any)
    s=Symbol(s)
    return @eval (($s) = ($v))
end

# ╔═╡ 27ad3673-7376-4db1-8a04-296d6e01c7ca
begin
	results = Pickle.npyload("mom_broad_q2_dep_more_events.pickle")
	q2s = results["q2s"]
	quarks = results["quarks"]
end

# ╔═╡ 18644342-68e0-4a6b-930b-48ac7050dde0
begin
	set_theme!(fonts = (; regular ="CMU Serif"))
	segmented_cmap = cgrad(:dense, length(q2s), categorical = true)
		
	q₂_labels = [L"%$q2ᵢ" for q2ᵢ in q2s]
	
	fig = Figure(resolution = (1000, 600), font = "CMU Serif")
	# fig = Figure(resolution = (600, 800), font = "CMU Serif")
	titles = [L"m\rightarrow\infty", L"m=m_\mathrm{beauty}", L"m=m_\mathrm{charm}"]
	ylabels = [L"\langle\delta p^2_T\rangle\,\mathrm{[GeV^2]}", L"\langle\delta p^2_L\rangle\,\mathrm{[GeV^2]}", L"\mathrm{ratio}"]
	# axes = [Axis(fig[j, i], 
	# 	xlabel=L"\delta\tau\,\mathrm{[fm/}c\mathrm{]}", ylabel=ylabels[j], xlabelsize = 22, ylabelsize= 22, xticklabelsize=16, yticklabelsize=16, xtickalign = 1, xticksize=4, ytickalign=1, yticksize=4, title= j == 1 ? titles[i] : "", titlesize=22
	# ) for i in 1:3, j in 1:3]
	axes = [Axis(fig[j, i], 
		xlabel=L"\delta\tau\,\mathrm{[fm/}c\mathrm{]}", ylabel=ylabels[j], xlabelsize = 22, ylabelsize= 22, xticklabelsize=16, yticklabelsize=16, xtickalign = 1, xticksize=4, ytickalign=1, yticksize=4, title= j == 1 ? titles[i] : "", titlesize=22
	# ) for i in 1:3, j in 1:3]
	) for i in 1:3, j in 1:2]

	for (iq, quark) in enumerate(quarks)
		# new results will start from τ=0
		τ = results["tau"][quark].-results["tau"][quark][1]
	
		for (i, q2) in enumerate(q2s)
			if (q2==4/3) || (q2==4)
				custom_linewidth = 4
			else
				custom_linewidth = 1.5
			end
			
			δp² = results["psq"][quark][string(q2)]
			lines!(axes[iq, 1], τ, (δp²[:,1]+δp²[:,2]), color=segmented_cmap[i], linewidth=custom_linewidth)
			lines!(axes[iq, 2], τ, δp²[:,3], color=segmented_cmap[i], linewidth=custom_linewidth)
		end
	
		δp²_qm = results["psq"][quark][string(4/3)]
		δp²_qm_T, δp²_qm_L = δp²_qm[:,1]+δp²_qm[:,2], δp²_qm[:,3]
		δp²_cl = results["psq"][quark][string(4.0)]./3
		δp²_cl_T, δp²_cl_L = δp²_cl[:,1]+δp²_cl[:,2], δp²_cl[:,3]
		ratio = δp²_cl./δp²_qm
	
		# lines!(axes[iq, 3], τ, δp²_cl_T./δp²_qm_T, color=segmented_cmap[5], linewidth=2)
		# lines!(axes[iq, 3], τ, δp²_cl_L./δp²_qm_L, color=segmented_cmap[13], linewidth=2)

		# ylims!(axes[iq, 3], 0, 2)
	end

	# rowsize!(fig.layout, 1, Relative(2/5))
	# rowsize!(fig.layout, 2, Relative(2/5))
	# rowsize!(fig.layout, 3, Relative(1/5))
	
	for i in 1:3
		# for j in 1:3
		for j in 1:2
			xlims!(axes[i, j], 0, 1.5)
			# if j!=3
			if j!=2	
				hidexdecorations!(axes[i, j], ticks = false, grid = false, ticklabels = false)
			end

			if i!=1	
				hideydecorations!(axes[i, j], ticks = false, grid = false, ticklabels = false)
			end
		end
	end

	# cbar = Colorbar(fig[1:3, 4], limits = (1, length(q2s)+1), colormap = segmented_cmap, size = 25, labelsize = 22, width = 10, flipaxis = true, ticksize=3, tickalign = 0, ticklabelsize = 14, height = Relative(1), label=L"q_2")
	cbar = Colorbar(fig[1:2, 4], limits = (1, length(q2s)+1), colormap = segmented_cmap, size = 25, labelsize = 22, width = 10, flipaxis = true, ticksize=3, tickalign = 0, ticklabelsize = 14, height = Relative(1), label=L"q_2")
	labels_q2 = ["0", "1/3", "2/3", "1", "4/3", "5/3", "2", "7/3", "8/3", "3", "10/3", "11/3", "4", "13/3", "14/3", "5", "16/3", "17/3", "6"]
	cbar.ticks = (range(1.5, 1.5+length(q2s)-1), labels_q2)

	labels_legend = [L"T", L"L"]
	elements= [LineElement(color = segmented_cmap[5], linewidth=2), LineElement(color = segmented_cmap[13], linewidth=2)]
	# axislegend(axes[1, 3], elements, labels_legend, position = :lt, labelsize=16, titlesize=20, orientation = :horizontal)

	save("plots/mom_broad_q2_dep_more_events_v2.png", fig, px_per_unit = 5.0)
	fig
end

# ╔═╡ 8ebf423b-0882-4555-ac10-f83a60eed7db
begin
	set_theme!(fonts = (; regular ="CMU Serif"))
	
	# fig_scaled = Figure(resolution = (1000, 800), font = "CMU Serif")

	ylabels_scaled = [L"\langle\delta p^2_T\rangle/q_2\,\mathrm{[GeV^2]}", L"\langle\delta p^2_L\rangle/q_2\,\mathrm{[GeV^2]}", L"\mathrm{ratio}"]
	fig_scaled = Figure(resolution = (1000, 600), font = "CMU Serif")
	axes_scaled = [Axis(fig_scaled[j, i], 
		xlabel=L"\delta\tau\,\mathrm{[fm/}c\mathrm{]}", ylabel=ylabels_scaled[j], xlabelsize = 22, ylabelsize= 22, xticklabelsize=16, yticklabelsize=16, xtickalign = 1, xticksize=4, ytickalign=1, yticksize=4, title= j == 1 ? titles[i] : "", titlesize=22
	# ) for i in 1:3, j in 1:3]
	) for i in 1:3, j in 1:2]

	for (iq, quark) in enumerate(quarks)
		# new results will start from τ=0
		τ = results["tau"][quark].-results["tau"][quark][1]
	
		for (i, q2) in enumerate(q2s)
			if (q2==4/3) || (q2==4)
				custom_linewidth = 4
			else
				custom_linewidth = 1.5
			end
			
			δp² = results["psq"][quark][string(q2)]./q2
			lines!(axes_scaled[iq, 1], τ, (δp²[:,1]+δp²[:,2]), color=segmented_cmap[i], linewidth=custom_linewidth)
			lines!(axes_scaled[iq, 2], τ, δp²[:,3], color=segmented_cmap[i], linewidth=custom_linewidth)
		end
	
		δp²_qm = results["psq"][quark][string(4/3)]
		δp²_qm_T, δp²_qm_L = δp²_qm[:,1]+δp²_qm[:,2], δp²_qm[:,3]
		δp²_cl = results["psq"][quark][string(4.0)]./3
		δp²_cl_T, δp²_cl_L = δp²_cl[:,1]+δp²_cl[:,2], δp²_cl[:,3]
		ratio = δp²_cl./δp²_qm
	
		# lines!(axes_scaled[iq, 3], τ, δp²_cl_T./δp²_qm_T, color=segmented_cmap[5], linewidth=2)
		# lines!(axes_scaled[iq, 3], τ, δp²_cl_L./δp²_qm_L, color=segmented_cmap[13], linewidth=2)

		# ylims!(axes_scaled[iq, 3], 0, 2)
	end

	# rowsize!(fig_scaled.layout, 1, Relative(2/5))
	# rowsize!(fig_scaled.layout, 2, Relative(2/5))
	# rowsize!(fig_scaled.layout, 3, Relative(1/5))
	
	for i in 1:3
		# for j in 1:3
		for j in 1:2
			xlims!(axes_scaled[i, j], 0, 1.5)
			# if j!=3
			if j!=2
				hidexdecorations!(axes_scaled[i, j], ticks = false, grid = false, ticklabels = false)
			end

			if i!=1	
				hideydecorations!(axes_scaled[i, j], ticks = false, grid = false, ticklabels = false)
			end
		end
	end

	# cbar_scaled = Colorbar(fig_scaled[1:3, 4], limits = (1, length(q2s)+1), colormap = segmented_cmap, size = 25, labelsize = 22, width = 10, flipaxis = true, ticksize=3, tickalign = 0, ticklabelsize = 14, height = Relative(1), label=L"q_2")
	cbar_scaled = Colorbar(fig_scaled[1:2, 4], limits = (1, length(q2s)+1), colormap = segmented_cmap, size = 25, labelsize = 22, width = 10, flipaxis = true, ticksize=3, tickalign = 0, ticklabelsize = 14, height = Relative(1), label=L"q_2")
	cbar_scaled.ticks = (range(1.5, 1.5+length(q2s)-1), labels_q2)

	# axislegend(axes_scaled[1, 3], elements, labels_legend, position = :lt, labelsize=16, titlesize=20, orientation = :horizontal)

	save("plots/mom_broad_div_q2_q2_dep_more_events_v2.png", fig_scaled, px_per_unit = 5.0)
	fig_scaled
end

# ╔═╡ 825d42aa-3df3-4f49-ab7f-7d9be1e51f3c
md"---
Casimir scaling"

# ╔═╡ 88330b49-fc34-4ef8-8bc6-a7d18e3bd76d
begin
	results_cs = Pickle.npyload("mom_broad_casimir_scaling.pickle")
	q2s_cs = results_cs["q2s"]
end

# ╔═╡ d4875c64-c9ed-4192-9020-93d41c0ce402
begin
	set_theme!(fonts = (; regular ="CMU Serif"))
	
	fig_cs = Figure(resolution = (1000, 350), font = "CMU Serif")
	# titles = [L"m\rightarrow\infty", L"m=m_\mathrm{beauty}", L"m=m_\mathrm{charm}"]
	axes_cs = [Axis(fig_cs[1, i], 
		xlabel=L"\delta\tau\,\mathrm{[fm/}c\mathrm{]}", ylabel=L"\langle\delta p^2\rangle_F/\langle\delta p^2\rangle_A", xlabelsize = 22, ylabelsize= 22, xticklabelsize=16, yticklabelsize=16, xtickalign = 1, xticksize=4, ytickalign=1, yticksize=4, title=titles[i], titlesize=22
	) for i in 1:3]

	for (iq, quark) in enumerate(quarks)
		# new results will start from τ=0
		τ = results["tau"][quark].-results["tau"][quark][1]

		δp²_adj = results["psq"][quark][string(3.0)]
		δp²_fund = results["psq"][quark][string(4/3)]

		lines!(axes_cs[iq], τ, (δp²_fund[:,1]+δp²_fund[:,2])./(δp²_adj[:,1]+δp²_adj[:,2]), color=segmented_cmap[5], linewidth=2)
		lines!(axes_cs[iq], τ, δp²_fund[:,3]./δp²_adj[:,3], color=segmented_cmap[13], linewidth=2)

		ylims!(axes_cs[iq], 0, 2*4/9)
		xlims!(axes_cs[iq], 0, 1.5)

		lines!(axes_cs[iq], τ, ones(length(τ)).*4/9, color=:gray, linewidth=2, linestyle=:dash)
		lines!(axes_cs[iq], τ, ones(length(τ)).*4/9, color=(:gray, 0.4), linewidth=2)

		# axes_cs[iq].yticks = ([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"])
	end

	for i in 2:3
		hideydecorations!(axes_cs[i], ticks = false, grid = false, ticklabels = false)
	end

	labels_legend_cs = [L"C_2\,(F\,)/C_2\,(A)=4/9"]
	elements_cs = [[LineElement(linewidth=2, color=:gray, linestyle=:dash), LineElement(linewidth=2, color=(:gray, 0.4))]]
	axislegend(axes_cs[1], elements, labels_legend, position = :lt, labelsize=16, titlesize=20, orientation = :horizontal)
	axislegend(axes_cs[2], elements_cs, labels_legend_cs, position = :lt, labelsize=16, titlesize=20, orientation = :horizontal)

	save("plots/mom_broad_casimir_scaling_v2.png", fig_cs, px_per_unit = 5.0)
	fig_cs
end

# ╔═╡ Cell order:
# ╠═79fb48c0-e377-11ed-31c1-45a5ae0977dc
# ╠═922896f4-1be4-4046-945c-51d3f39f30dd
# ╠═27ad3673-7376-4db1-8a04-296d6e01c7ca
# ╠═18644342-68e0-4a6b-930b-48ac7050dde0
# ╠═8ebf423b-0882-4555-ac10-f83a60eed7db
# ╠═825d42aa-3df3-4f49-ab7f-7d9be1e51f3c
# ╠═88330b49-fc34-4ef8-8bc6-a7d18e3bd76d
# ╠═d4875c64-c9ed-4192-9020-93d41c0ce402
