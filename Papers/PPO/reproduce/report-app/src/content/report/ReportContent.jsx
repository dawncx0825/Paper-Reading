import React from "react";

import {
  DataComponent,
  DataTable,
  MetricCard,
  ReportSection,
  RichNarrative,
  useDataApp,
} from "../../data-app-public.jsx";

const percent = (value, digits = 2) => Number.isFinite(value)
  ? `${(value * 100).toFixed(digits)}%`
  : "—";

const fixed = (value, digits = 4) => Number.isFinite(value)
  ? value.toFixed(digits)
  : "—";

function EvidenceTable({ id, title, queryId, sourceRows, displayRows, columns, description }) {
  return <DataComponent id={id} title={title} queryId={queryId} kind="table"
    sourceRows={sourceRows} displayRows={displayRows} description={description}>
    <DataTable rows={displayRows} columns={columns} searchable={false}
      compactNumbers={false} pageSize={displayRows.length || 1} caption={title} />
  </DataComponent>;
}

export function ReportContent() {
  const { reviewedPeriodRows, visible, canEdit, mode, appTitle, setAppTitle } = useDataApp();
  const training = reviewedPeriodRows("training_stages");
  const rm = reviewedPeriodRows("rm_evaluation");
  const automatic = reviewedPeriodRows("automatic_metrics");
  const judge = reviewedPeriodRows("judge_results");
  const validation = reviewedPeriodRows("validation");

  const ppoVsSft = judge.find((row) => row.comparison === "PPO vs SFT");
  const rmTest = rm.find((row) => row.split === "Test");
  const ppoStage = training.find((row) => row.stage === "PPO");

  const trainingDisplay = training.map((row) => ({
    stage: row.stage,
    scale: row.trainingScale,
    wallTime: row.wallLabel,
    peakAllocated: `${row.peakAllocatedGiB.toFixed(2)} GiB`,
    peakReserved: `${row.peakReservedGiB.toFixed(2)} GiB`,
    result: `${row.primaryMetric}: ${fixed(row.primaryValue, 4)}`,
  }));

  const rmDisplay = rm.map((row) => ({
    split: row.split,
    coverage: `${row.pairs.toLocaleString()} pairs / ${row.posts} posts`,
    accuracy: percent(row.accuracy),
    confidenceInterval: `[${percent(row.ciLow)}, ${percent(row.ciHigh)}]`,
    meanMargin: fixed(row.meanMargin, 4),
    lengthCorrelation: fixed(row.rewardLengthCorrelation, 4),
  }));

  const metricDisplay = automatic.map((row) => ({
    model: row.model,
    rouge1: fixed(row.rouge1, 4),
    rouge2: fixed(row.rouge2, 4),
    rougeL: fixed(row.rougeL, 4),
    meanWords: row.meanWords.toFixed(2),
    repetition: percent(row.repetition3gram),
    eosRate: percent(row.eosRate),
  }));

  const judgeDisplay = judge.map((row) => ({
    comparison: row.comparison,
    target: row.target,
    completePosts: row.completePosts,
    preference: percent(row.preferenceScore),
    confidenceInterval: `[${percent(row.ciLow)}, ${percent(row.ciHigh)}]`,
    orderConsistency: percent(row.orderConsistency),
  }));

  return <article className="report-content" aria-label="PPO 论文复现最终结果报告">
    <header className="report-hero">
      <h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning
        aria-label={canEdit && mode === "edit" ? "编辑报告标题" : undefined}
        onBlur={canEdit && mode === "edit" ? (event) => setAppTitle(event.currentTarget.textContent.trim() || appTitle) : undefined}
        onKeyDown={canEdit && mode === "edit" ? (event) => {
          if (event.key === "Enter") { event.preventDefault(); event.currentTarget.blur(); }
        } : undefined}>{appTitle}</h1>
      <RichNarrative id="report:description" className="report-deck" label="编辑报告说明"
        value="单张 RTX 3090 上完成 Base → SFT → Reward Model → PPO 的端到端复现，并在固定测试集上执行自动指标与 Qwen 双顺序盲评。以下所有数字均来自已同步的正式运行日志和结果文件。" />
    </header>

    {visible("executive-summary") && <ReportSection id="executive-summary" title="执行摘要"
      queryId="judge_results" sourceRows={judge} showHeading={false} className="report-summary">
      <RichNarrative id="executive-summary:body" className="report-summary-lead" label="编辑执行摘要"
        value={`## 执行摘要\n\n- 工程复现成功：完整流水线退出码为 0，8 项测试通过，PPO 完成全部 10,240 episodes。\n- 核心偏好结果：PPO 相对 SFT 的 Qwen 裁判偏好分数为 ${percent(ppoVsSft?.preferenceScore)}，95% CI 为 [${percent(ppoVsSft?.ciLow)}, ${percent(ppoVsSft?.ciHigh)}]。\n- 关键限制：PPO 的 ROUGE-L 低于 SFT，且 RM 测试集 reward–length correlation 为 ${fixed(rmTest?.rewardLengthCorrelation, 3)}；偏好优势可能部分来自更长输出。`} />
    </ReportSection>}

    <div className="report-facts" aria-label="关键结果">
      {visible("metric-ppo-preference") && <MetricCard id="metric-ppo-preference" title="PPO 对 SFT 偏好分数"
        queryId="judge_results" sourceRows={judge} value={percent(ppoVsSft?.preferenceScore)}
        comparison="95% CI 68.56%–75.10%" description="Qwen3-Max 双顺序盲评；平局计 0.5。" />}
      {visible("metric-rm-accuracy") && <MetricCard id="metric-rm-accuracy" title="RM 测试准确率"
        queryId="rm_evaluation" sourceRows={rm} value={percent(rmTest?.accuracy)}
        comparison="95% CI 60.14%–68.11%" description="按来源帖子分组 bootstrap。" />}
      {visible("metric-ppo-runtime") && <MetricCard id="metric-ppo-runtime" title="PPO 墙钟时间"
        queryId="training_stages" sourceRows={training} value={ppoStage?.wallLabel ?? "—"}
        comparison="峰值 reserved 10.61 GiB" description="单张 RTX 3090，320 个 rollout batches。" />}
    </div>

    <section className="report-section">
      <RichNarrative id="training:heading" className="report-analysis" label="编辑训练结果说明"
        value="## 训练耗时与显存\n\n三个训练阶段均在单张 RTX 3090 上完成，没有出现 OOM 或 NaN。PPO 是主要耗时阶段，占三阶段训练总时间约 86%。" />
      {visible("training-table") && <EvidenceTable id="training-table" title="正式训练阶段"
        queryId="training_stages" sourceRows={training} displayRows={trainingDisplay}
        description="正式 full profile 的阶段规模、墙钟时间、PyTorch CUDA 峰值和主要结果。"
        columns={[
          { field: "stage", label: "阶段", presentation: "identity" },
          { field: "scale", label: "训练量" },
          { field: "wallTime", label: "墙钟时间" },
          { field: "peakAllocated", label: "峰值 allocated" },
          { field: "peakReserved", label: "峰值 reserved" },
          { field: "result", label: "主要结果" },
        ]} />}
    </section>

    <section className="report-section">
      <RichNarrative id="rm:heading" className="report-analysis" label="编辑奖励模型说明"
        value="## Reward Model 独立评测\n\nRM 在独立测试集上显著高于随机，但奖励差与候选长度差有较强正相关。这说明它提供了可用偏好信号，也可能系统性鼓励较长摘要。" />
      {visible("rm-table") && <EvidenceTable id="rm-table" title="Reward Model 评测"
        queryId="rm_evaluation" sourceRows={rm} displayRows={rmDisplay}
        description="准确率区间按来源帖子分组 bootstrap；长度相关系数为 Pearson correlation。"
        columns={[
          { field: "split", label: "Split", presentation: "identity" },
          { field: "coverage", label: "覆盖" },
          { field: "accuracy", label: "偏好准确率" },
          { field: "confidenceInterval", label: "95% CI" },
          { field: "meanMargin", label: "平均 margin" },
          { field: "lengthCorrelation", label: "Reward–长度相关" },
        ]} />}
    </section>

    <section className="report-section">
      <RichNarrative id="metrics:heading" className="report-analysis" label="编辑自动指标说明"
        value="## 固定测试集自动指标\n\nSFT 相对 Base 大幅改善 ROUGE、终止率和重复控制。PPO 的 ROUGE-1 接近 SFT，但 ROUGE-2 与 ROUGE-L 略低；其平均输出比 SFT 长约 20.7 词。" />
      {visible("metrics-table") && <EvidenceTable id="metrics-table" title="500 个隔离测试帖的自动指标"
        queryId="automatic_metrics" sourceRows={automatic} displayRows={metricDisplay}
        description="三个模型使用相同测试帖子和生成设置；重复率越低越好，EOS 率越高越好。"
        columns={[
          { field: "model", label: "模型", presentation: "identity" },
          { field: "rouge1", label: "ROUGE-1" },
          { field: "rouge2", label: "ROUGE-2" },
          { field: "rougeL", label: "ROUGE-L" },
          { field: "meanWords", label: "平均词数" },
          { field: "repetition", label: "3-gram 重复率 ↓" },
          { field: "eosRate", label: "EOS 率 ↑" },
        ]} />}
    </section>

    <section className="report-section">
      <RichNarrative id="judge:heading" className="report-analysis" label="编辑裁判结果说明"
        value="## Qwen 双顺序盲评\n\n每个摘要对按 A/B 与 B/A 两种顺序判断；先在帖子内平均，再按帖子 bootstrap 10,000 次。2,986/3,000 个唯一调用成功，14 个判断被内容安全过滤器稳定拒绝。" />
      {visible("judge-table") && <EvidenceTable id="judge-table" title="测试集成对偏好结果"
        queryId="judge_results" sourceRows={judge} displayRows={judgeDisplay}
        description="目标模型的帖子级偏好分数；平局贡献 0.5，仅统计具备双顺序完整结果的帖子。"
        columns={[
          { field: "comparison", label: "比较", presentation: "identity" },
          { field: "target", label: "目标模型" },
          { field: "completePosts", label: "完整帖子" },
          { field: "preference", label: "目标偏好分数" },
          { field: "confidenceInterval", label: "95% CI" },
          { field: "orderConsistency", label: "顺序一致率" },
        ]} />}
    </section>

    {visible("limitations") && <ReportSection id="limitations" title="结论与限制"
      queryId="judge_results" queryIds={["judge_results", "rm_evaluation", "automatic_metrics"]}
      sourceRowsByQuery={{ judge_results: judge, rm_evaluation: rm, automatic_metrics: automatic }}
      sourceRows={judge} showHeading={false} className="report-methods">
      <RichNarrative id="limitations:body" className="report-caveat" label="编辑结论与限制"
        value="## 结论与限制\n\n**最准确的结论是：完整工程复现成功，方法方向得到支持；但这不是原论文规模和人工评测条件下的绝对数值复刻。**\n\n- 本轮仅运行 1 个 seed，尚不能量化训练随机性。\n- Qwen 裁判偏好不能等同于人类偏好。\n- RM 的长度偏差明显，建议补做长度匹配评测、人工盲评和多 seed 实验。\n- PPO 的 ROUGE-L 低于 SFT，说明裁判偏好与参考文本重合指标之间存在张力。" />
    </ReportSection>}

    <section className="report-section">
      <RichNarrative id="validation:heading" className="report-analysis" label="编辑交付验证说明"
        value="## 交付与复核\n\n以下检查用于确认结果可追溯、代码路径可运行，并避免把大模型 checkpoint 下载到本地。" />
      {visible("validation-table") && <EvidenceTable id="validation-table" title="运行完整性检查"
        queryId="validation" sourceRows={validation} displayRows={validation}
        description="来自流水线 marker、数据准备日志和服务器测试输出。"
        columns={[
          { field: "check", label: "检查项", presentation: "identity" },
          { field: "result", label: "状态", presentation: "status" },
          { field: "detail", label: "证据" },
        ]} />}
    </section>
  </article>;
}
