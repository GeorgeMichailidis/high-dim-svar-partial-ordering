
rm(list=ls())

## https://bioconductor.org/packages/release/data/experiment/vignettes/DREAM4/inst/doc/DREAM4.pdf
require(DREAM4)
require(xlsx)

print(getwd())

## 'dream4_100_01', 'dream4_100_02', 'dream4_100_03', 'dream4_100_04', 'dream4_100_05'
filename = 'dream4_100_05'
data(dream4_100_05)
mtx.all = assays(dream4_100_05)[[1]]
mtx.goldStandard = metadata(dream4_100_05)[[1]]

## process data
data_list = list()
for (fold_id in 1:10)
{
    columns = grep(sprintf("perturbation.%s.",fold_id), colnames(mtx.all), fixed=TRUE)    
    mtx.ts = data.frame(t(mtx.all[, columns]))
    mtx.ts$fold_id = fold_id 
    
    data_list[[fold_id]] = mtx.ts
} 
df = do.call('rbind',data_list)

## process adjacency matrix
idx = which(mtx.goldStandard == 1)
idx.m1 = idx -1
rows = idx.m1 %% nrow (mtx.goldStandard) + 1
cols = idx.m1 %/% nrow (mtx.goldStandard) + 1
df.goldStandard = data.frame(Regulator=rownames(mtx.goldStandard)[rows], 
                             Target=colnames(mtx.goldStandard)[cols],
                             Source=rep('goldStandard', length(rows)), 
                             stringsAsFactors=FALSE)
A = data.frame(t(mtx.goldStandard))

## write to excel so that it can be read by python
write.xlsx(df, file=sprintf("%s.xlsx", filename), sheetName="data", row.names=TRUE)
write.xlsx(A, file=sprintf("%s.xlsx", filename), sheetName="A", append=TRUE, row.names=TRUE)
write.xlsx(df.goldStandard, file=sprintf("%s.xlsx", filename), sheetName="src_target", append=TRUE, row.names=TRUE)
