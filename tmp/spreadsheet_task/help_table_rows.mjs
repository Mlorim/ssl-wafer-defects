import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load("/Users/mironpomazkov/Desktop/ssl-wafer-defects/outputs/model_metrics_comparison.xlsx"));
console.log(workbook.help("*", { search: "table.*row|row.*delete|delete.*row", include: "index,examples,notes", maxChars: 8000 }).ndjson);
