#Drinking per bodyweight of drinkometer rats
#Libraries

library("readxl")
library("writexl")
library("combinat")

#get data
day1 = read_excel("Path/to/Drinking_Data_Day1.xlsx")
day2 = read_excel("Path/to/Drinking_Data_Day2.xlsx")

#clean data
day1 = day1[-c("Animal_x")] #remove animals if reasonably necessary 
day2 = day2[-c("Animal_x")] #remove animals if reasonably necessary


#format into df & numerization
day1 = as.data.frame(day1)
day2 = as.data.frame(day2)
cols <- c("Wasser", "5%ETOH", "10%ETOH", "20%ETOH")
day1[cols] <- lapply(day1[cols], as.numeric)
day2[cols] <- lapply(day2[cols], as.numeric)


#calculate drinking volume 
drinkvolume = as.matrix(day1) -  as.matrix(day2) #volume = bottleweight friday - monday
drinkvolume


#summarize into concrete alcohol amount
vol = data.frame("alcohol" = 
            drinkvolume[,3] /20 +
            drinkvolume[,4] /10 +
            drinkvolume[,5] /5
)

#per day
volpd = vol/3

#calculate bodyweight: Vol/Körpergewicht * 1000
volpdbdw = read_excel("Path/to/Drinking_volume_per_day.xlsx")
volpdbdw$`Vol/d` = volpd
volpdbdw$`Vol/d/kg` = volpdbdw$`Vol/d`*volpdbdw$Bodyweight / 1000


###unlist for saving
volpdbdw$`Vol/d` = unlist(volpdbdw$`Vol/d`)
volpdbdw$`Vol/d/kg` = unlist(volpdbdw$`Vol/d/kg`)
volpdbdw$`Vol/d/kg`= unname(volpdbdw$`Vol/d/kg`)
volpdbdw$`Vol/d`= unname(volpdbdw$`Vol/d`)


#saving
write_xlsx(volpdbdw, "Path/to/saving.xlsx")


#######Balanced group allocation
#Standardise mean and SD for comparison

volpdbdw$bw_z = scale(volpdbdw$Bodyweight)
volpdbdw$bl_z = scale(volpdbdw$`Vol/d/kg`)

m_volpdbw = volpdbdw["first male animal number: last male animal number",] #males
m_volpdbw$`Vol/d/kg`


#male
comb <- combn(nrow(m_volpdbw), nrow(m_volpdbw)/2) #makes all possible combinations of splitting the animals in 2 groups
var_weight = 0.7 #weighing of variance, may be altered freely; high value = SD heavier, low value = mean heavier

# 3) aimed function
objective <- function(idx) {
  gA <- m_volpdbw[idx, ]     #Group x
  gB <- m_volpdbw[-idx, ]   #Group y
  
  diff_bw_mean <- mean(gA$bw_z) - mean(gB$bw_z)   #calculate mean difference; closer to 0 = lower difference
  diff_bl_mean <- mean(gA$bl_z) - mean(gB$bl_z)
  
  diff_bw_sd   <- sd(gA$bw_z) - sd(gB$bw_z)       # differences in spread
  diff_bl_sd   <- sd(gA$bl_z) - sd(gB$bl_z)
  
  sqrt(diff_bw_mean^2 + diff_bl_mean^2 + var_weight*(diff_bw_sd^2 + diff_bl_sd^2))     #distance between groups; low score = low difference
}

# 4) find best combination
scores <- apply(comb, 2, objective)   # apply function onto every combination
best   <- comb[, which.min(scores)]   #"Filter": lowest score

# 5) group allocation
m_volpdbw$group <- "B"
m_volpdbw$group[best] <- "A"

aggregate(cbind(Bodyweight, `Vol/d/kg`) ~ group, m_volpdbw, mean)
aggregate(cbind(Bodyweight, `Vol/d/kg`) ~ group, m_volpdbw, sd)


###Plot
###Plot
boxplot(`Vol/d/kg` ~ group, data = m_volpdbw)
stripchart(`Vol/d/kg` ~ group,  data = m_volpdbw,  vertical = TRUE,  method = "jitter",  pch = 16,  add = TRUE)

boxplot(Bodyweight ~ group, data = m_volpdbw)
stripchart(Bodyweight ~ group, data = m_volpdbw, vertical = TRUE,  method = "jitter", pch = 16,  add = TRUE)

boxplot(`Vol/d/kg` ~ group, data = f_volpdbw)
stripchart(`Vol/d/kg` ~ group,  data = f_volpdbw,  vertical = TRUE,  method = "jitter",  pch = 16,  add = TRUE)

boxplot(Bodyweight ~ group, data = f_volpdbw)
stripchart(Bodyweight ~ group, data = f_volpdbw, vertical = TRUE,  method = "jitter", pch = 16,  add = TRUE)


##combine and clean
grouplist = rbind(f_volpdbw, m_volpdbw)

#save
write_xlsx(grouplist, "Path/to/saving.xlsx")


## Sort subgroups:
#- Male B into 2
#df only with animals in group "B"
sub_m = subset(grouplist, group == "B")
sub_m = subset(sub_m, Sex == "m")


#balance male
comb <- combn(nrow(sub_m), nrow(sub_m)/2)
objective <- function(idx) {
  gA <- sub_m[idx, ]
  gB <- sub_m[-idx, ]
  diff_bw_mean <- mean(gA$bw_z) - mean(gB$bw_z)
  diff_bl_mean <- mean(gA$bl_z) - mean(gB$bl_z)
  diff_bw_sd   <- sd(gA$bw_z) - sd(gB$bw_z)
  diff_bl_sd   <- sd(gA$bl_z) - sd(gB$bl_z)
  sqrt(diff_bw_mean^2 + diff_bl_mean^2 + var_weight*(diff_bw_sd^2 + diff_bl_sd^2))
}
scores <- apply(comb, 2, objective)
best   <- comb[, which.min(scores)]
sub_m$group <- "B"
sub_m$group[best] <- "C"
sub_m

#look at data
boxplot(`Vol/d/kg` ~ group, data = sub_m)
stripchart(`Vol/d/kg` ~ group,  data = sub_m,  vertical = TRUE,  method = "jitter",  pch = 16,  add = TRUE)
boxplot(Bodyweight ~ group, data = sub_m)
stripchart(Bodyweight ~ group, data = sub_m, vertical = TRUE,  method = "jitter", pch = 16,  add = TRUE)
aggregate(cbind(Bodyweight, `Vol/d/kg`) ~ group, sub_m, mean)
aggregate(cbind(Bodyweight, `Vol/d/kg`) ~ group, sub_m, sd)

##clean and save
grouplist = rbind(subset(grouplist, group == "A"),
                  sub_f, sub_m)
grouplist$bw_z = NULL
grouplist$bl_z = NULL

grouplist
write_xlsx(grouplist, "path/to/saving")
