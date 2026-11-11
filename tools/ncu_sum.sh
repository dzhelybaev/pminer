#!/usr/bin/env bash
# Короткая выжимка отчётов tools/ncu.sh: причины простоев warp, загрузка блоков SM, память.
#   bash ncu_sum.sh [GPU...]   (по умолчанию 0 1 2)
D=/root/ncu
NCU=$(ls "$D"/opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)
[ -n "$NCU" ] || { echo "ncu не найден — сначала tools/ncu.sh"; exit 1; }
M='gpu__time_duration.sum,smsp__issue_active.avg.pct_of_peak_sustained_active'
M="$M,smsp__warps_eligible.avg.per_cycle_active,dram__throughput.avg.pct_of_peak_sustained_elapsed"
M="$M,lts__t_sector_hit_rate.pct,lts__throughput.avg.pct_of_peak_sustained_elapsed"
M="$M,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"
M="$M,regex:sm__pipe_.*_cycles_active.avg.pct_of_peak_sustained_active"
M="$M,regex:sm__inst_executed_pipe_.*.avg.pct_of_peak_sustained_active"
M="$M,regex:smsp__average_warps_issue_stalled_.*_per_issue_active.ratio"
for g in ${@:-0 1 2}; do
  [ -f "$D/gpu$g.ncu-rep" ] || { echo "== GPU$g: нет отчёта"; continue; }
  echo "== GPU$g"
  "$NCU" -i "$D/gpu$g.ncu-rep" --page raw --csv --metrics "$M" 2>/dev/null |
  awk -F'","' 'NR==1{for(i=1;i<=NF;i++){gsub(/"/,"",$i);n[i]=$i}}
    NR==3{for(i=1;i<=NF;i++){gsub(/"/,"",$i);gsub(/,/,"",$i);v=$i+0;
      if(n[i]!~/^(gpu__|smsp__|dram__|lts__|l1tex__|sm__)/)continue;
      if(n[i]~/stalled|pipe_/ && v<0.05)continue;
      s=n[i];sub(/smsp__average_warps_issue_stalled_/,"stall ",s);sub(/_per_issue_active.ratio/,"",s);
      sub(/.avg.pct_of_peak_sustained_(active|elapsed)/," %",s);printf "  %-60s %s\n",s,$i}}' |
  cat
done
