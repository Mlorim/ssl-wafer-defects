import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir =
  "/Users/mironpomazkov/Desktop/ssl-wafer-defects/outputs/019f957c-8bac-71a2-8c81-857e4eff9795";
const outputPath = `${outputDir}/model_metrics_comparison.xlsx`;

const rows = [
  ["Baseline (ResNet-18)", "WM-811K; 10% labeled; SMOTE; random 80/20 split", 0.7917, 0.7956, 0.7899, 0.7887, 0.9496964441, 0.8341579926, 0.6459195682, 0.7059864431, "Текущий результат получен всего за 2 эпохи; отличаются разбиение данных, SMOTE и предобработка. Высокая общая доля правильных ответов скрывает низкую полноту редких классов."],
  ["Mean Teacher", "WM-811K; 10% labeled + unlabeled; SMOTE; random 80/20 split", 0.8114, 0.8255, 0.8115, 0.8129, 0.9423532813, 0.6489157538, 0.7596512398, 0.6899986630, "Всего 2 эпохи; результат чувствителен к EMA, весу функции согласованности, аугментациям и составу пула неразмеченных данных. Возможны различия в расчёте макроусреднённых метрик."],
  ["SupCon", "WM-811K; 10% labeled; SMOTE; random 80/20 split", 0.8413, 0.8553, 0.8337, 0.8298, 0.9540907777, 0.7386761339, 0.7881932752, 0.7512159581, "Всего 2 эпохи; контрастивное представление обычно требует более долгого обучения и подбора температуры и аугментаций."],
  ["Mean Teacher + SupCon", "WM-811K; 10% labeled + unlabeled; SMOTE; random 80/20 split", 0.8463, 0.8624, 0.8441, 0.8340, 0.9575310783, 0.7192984243, 0.7743321524, 0.7213993019, "Всего 2 эпохи; одновременно не настроены компоненты полуконтролируемого и контрастивного обучения. Отличаются разбиение, балансировка и протокол расчёта метрик."],
  ["VAE latent-vector representation", "WM-811K; 10% labeled; SMOTE; ResNet-50 teacher/student", 0.9770, 0.9460, 0.9120, 0.9620, 0.9668979474, 0.8804275468, 0.7948880620, 0.8314339052, "Статья не полностью задаёт расписание VAE и псевдоразметки; результат чувствителен к top-K, качеству учителя и дообучению. Возможно другое разбиение или усреднение метрик."],
  ["CBAM-CNN", "WM-811K subset только native 26×26; train-only CAE augmentation", 0.9988, 0.9956, 0.9900, 0.9983, 0.9024390244, 0.4578177412, 0.5500783424, 0.4785992066, "Очень малая и крайне редкая подвыборка карт исходного размера 26×26; честное разбиение оставляет единицы образцов некоторых классов. В статье вероятна утечка через CAE-аугментацию или использована другая выборка."],
  ["SemiWaferNet Hybrid CNN–ViT", "WM-811K official split; 32×32; balanced labeled + 150k unlabeled", 0.9872, null, null, 0.9861, 0.7536489734, 0.5232008921, 0.6061249993, 0.5087141718, "Слабый исходный учитель и ошибки псевдоразметки накапливаются по стадиям; не описаны пороги и расписание. Возможны другая официальная подвыборка и утечка при повторной выборке данных."],
  ["MM-WAE", "WM-811K; leakage-safe stratified 70/10/20; 10% labeled; 32×32", 0.9712, 0.9611, 0.9623, 0.9617, 0.9542642382, 0.7795622346, 0.7112460508, 0.7384687412, "Accuracy близка, но макроусреднённые метрики проседают на Scratch и Loc; результат чувствителен к весам функций потерь, ширине ядра MMD, весам классов и разбиению."],
  ["MobileNetV3-Small (Efficient CNN)", "WM-811K labeled; leakage-safe 80/20 + 4-fold; train-only oversampling; 224×224", 0.9800, 0.9090, 0.8850, 0.8950, 0.9621566927, 0.8219448248, 0.8534369599, 0.8324200938, "Статья не указывает использование предобученных весов, число эпох, планировщик скорости обучения и параметры аугментаций. Возможна утечка из-за аугментации до разбиения; основной разрыв — в precision редких классов."],
];

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("Сравнение моделей");
sheet.showGridLines = false;

sheet.getRange("A1:O1").merge();
sheet.getRange("A1").values = [["Сравнение заявленных и воспроизведённых метрик"]];
sheet.getRange("A1:O1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
sheet.getRange("A1:O1").format.rowHeight = 30;

sheet.getRange("A2:O2").merge();
sheet.getRange("A2").values = [[
  "Все метрики — доли от 0 до 1; Δ = мой результат − результат авторов. Отрицательное Δ означает отставание."
]];
sheet.getRange("A2:O2").format = {
  fill: "#D9EAF7",
  font: { color: "#17365D", italic: true },
  horizontalAlignment: "left",
  verticalAlignment: "center",
};
sheet.getRange("A2:O2").format.rowHeight = 24;

const headers = [
  "Модель", "Датасет / протокол",
  "Авторы: Accuracy", "Авторы: Precision", "Авторы: Recall", "Авторы: F1",
  "Мой: Accuracy", "Мой: Precision", "Мой: Recall", "Мой: F1",
  "Δ Accuracy", "Δ Precision", "Δ Recall", "Δ F1",
  "Предполагаемые причины расхождения",
];
sheet.getRange("A4:O4").values = [headers];
sheet.getRange("A4:O4").format = {
  fill: "#4472C4",
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "outside", style: "medium", color: "#2F5597" },
};
sheet.getRange("A4:O4").format.rowHeight = 42;

const valueRows = rows.map((row) => [
  ...row.slice(0, 10),
  null, null, null, null,
  row[10],
]);
sheet.getRange(`A5:O${4 + rows.length}`).values = valueRows;

for (let i = 0; i < rows.length; i += 1) {
  const r = 5 + i;
  sheet.getRange(`K${r}:N${r}`).formulas = [[
    `=IF(OR(C${r}="",G${r}=""),"",G${r}-C${r})`,
    `=IF(OR(D${r}="",H${r}=""),"",H${r}-D${r})`,
    `=IF(OR(E${r}="",I${r}=""),"",I${r}-E${r})`,
    `=IF(OR(F${r}="",J${r}=""),"",J${r}-F${r})`,
  ]];
}

const dataRange = sheet.getRange(`A5:O${4 + rows.length}`);
dataRange.format = {
  verticalAlignment: "top",
  borders: {
    insideHorizontal: { style: "thin", color: "#D9E2F3" },
    bottom: { style: "thin", color: "#A6A6A6" },
  },
};
sheet.getRange(`A5:B${4 + rows.length}`).format.wrapText = true;
sheet.getRange(`O5:O${4 + rows.length}`).format.wrapText = true;
sheet.getRange(`C5:N${4 + rows.length}`).format = {
  numberFormat: "0.00%",
  horizontalAlignment: "right",
  verticalAlignment: "top",
};
sheet.getRange(`K5:N${4 + rows.length}`).conditionalFormats.add("colorScale", {
  colors: ["#F8696B", "#FFEB84", "#63BE7B"],
  thresholds: ["min", { type: "num", value: 0 }, "max"],
});

for (let i = 0; i < rows.length; i += 1) {
  if (i % 2 === 1) {
    sheet.getRange(`A${5 + i}:O${5 + i}`).format.fill = "#F4F7FB";
  }
}

sheet.getRange("A:A").format.columnWidth = 29;
sheet.getRange("B:B").format.columnWidth = 42;
sheet.getRange("C:N").format.columnWidth = 16;
sheet.getRange("O:O").format.columnWidth = 58;
sheet.getRange(`A5:O${4 + rows.length}`).format.rowHeight = 66;
sheet.freezePanes.freezeRows(4);
sheet.freezePanes.freezeColumns(2);

const table = sheet.tables.add(`A4:O${4 + rows.length}`, true, "ModelMetricsComparison");
table.style = "TableStyleMedium2";
table.showFilterButton = true;

const noteRow = 6 + rows.length;
sheet.getRange(`A${noteRow}:O${noteRow}`).merge();
sheet.getRange(`A${noteRow}`).values = [[
  "Примечание: smoke-тесты исключены. Для Mean Teacher/SupCon в текущем full-файле фактически сохранён прогон на 2 эпохи, поэтому эти строки нельзя считать окончательной репликацией статьи."
]];
sheet.getRange(`A${noteRow}:O${noteRow}`).format = {
  fill: "#FFF2CC",
  font: { italic: true, color: "#7F6000" },
  wrapText: true,
  verticalAlignment: "center",
};
sheet.getRange(`A${noteRow}:O${noteRow}`).format.rowHeight = 36;

await fs.mkdir(outputDir, { recursive: true });
const preview = await workbook.render({
  sheetName: "Сравнение моделей",
  range: `A1:O${noteRow}`,
  scale: 1,
  format: "png",
});
await fs.writeFile(`${outputDir}/model_metrics_comparison_preview.png`, new Uint8Array(await preview.arrayBuffer()));

const inspection = await workbook.inspect({
  kind: "table",
  range: `Сравнение моделей!A1:O${noteRow}`,
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 15,
});
console.log(inspection.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(outputPath);
