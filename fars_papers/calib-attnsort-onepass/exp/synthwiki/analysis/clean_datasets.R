# Get the result files
joinCSVFiles <- function(directory) {
  setwd(directory)
  f <- list.files()
  f <- f[grep('csv', f)]
  list_of_dataframes <- lapply(f, function(z) read.csv(z, stringsAsFactors = FALSE))
  df <- do.call(rbind, list_of_dataframes)
  return(df)
}

dirs <- c('~/Dropbox/Cut Attention/Experiments/smoothbrain_madlibs1_28k_random',
          '~/Dropbox/Cut Attention/Experiments/smoothbrain_madlibs1_12k_random',
          '~/Dropbox/Cut Attention/Experiments/smoothbrain_madlibs1_6k_random',
          '~/Dropbox/Cut Attention/Experiments/smoothbrain_madlibs1_3k_random',
          '~/Dropbox/Cut Attention/Experiments/smoothbrain_madlibs1_1k_random')

all_data <- data.frame()
for (d in dirs) {
  print(d)
  tmp <- joinCSVFiles(d)
  all_data <- rbind(all_data, tmp)
}
all_data <- all_data %>% group_by(junk_size) %>% mutate(min_num_docs = min(total_docs))
all_data <- all_data %>% filter(smoothing < min_num_docs)

all_data$Method <- paste("Truncaction k = ", all_data$smoothing, sep='')
all_data$Method[all_data$smoothing == 5] <- "Truncaction k = 05"

all_data$Method[all_data$smoothing == -1] <- 'Full Context'

setwd("~/Documents/cut_attention/analysis")
write.csv(all_data, 'llama_acc_madlib1.csv')

all_data %>%
  #filter(smoothing != 50) %>%
  group_by(junk_size, Method) %>%
  summarise(m = mean(correct),
            sem = sd(correct) / sqrt(n())) %>%
  ggplot(data=., aes(x=junk_size, y=m, group=Method, 
                     colour=Method)) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(ymin = m-sem*1.96, ymax = m+sem*1.96), width=0) +
  theme_bw() + 
  xlab("Context Size (Tokens)") +
  ylab("QA Accuracy")

# Get the attention files

dirs <- c('~/Dropbox/Cut Attention/Experiments/smoothbrain_madlibs1_28k_random/citations',
          '~/Dropbox/Cut Attention/Experiments/smoothbrain_madlibs1_12k_random/citations',
          '~/Dropbox/Cut Attention/Experiments/smoothbrain_madlibs1_6k_random/citations',
          '~/Dropbox/Cut Attention/Experiments/smoothbrain_madlibs1_3k_random/citations',
          '~/Dropbox/Cut Attention/Experiments/smoothbrain_madlibs1_1k_random/citations')

attn_data <- data.frame()
for (d in dirs) {
  tmp <- joinCSVFiles(d)
  attn_data <- rbind(attn_data, tmp)
}

attn_data <- attn_data %>%
  filter(generation_token == 0) %>%
  mutate(Document = ifelse(doc == true_doc_position, 'Correct Doc', 'Distractor')) 


setwd("~/Documents/cut_attention/analysis")
write.csv(all_data, 'llama_attn_madlib1.csv')

attn_data$ctx <- paste(attn_data$junk_size, " in Context", sep = '')

# Create the factor with the desired order
attn_data$ctx <- factor(attn_data$ctx, 
                        levels = rev(unique(paste(attn_data$junk_size, " in Context", sep = ''))))

attn_data %>%
  group_by(Document, layer, ctx) %>%
  summarise(m = mean(attn_sum)) %>%
  ggplot(data=., aes(x=layer, y=m, colour=Document, group=Document)) +
  geom_line() +
  facet_grid(~ctx) +
  theme_bw() +
  xlab("Layer") +
  ylab("Mean Attention") +
  theme(legend.position='bottom')


# Get the GPT files

dirs <- c('~/Dropbox/Cut Attention/Experiments/madlibs1_gpt3_10k_qa/',
          '~/Dropbox/Cut Attention/Experiments/madlibs1_gpt3_1k_qa/')
gpt_qa <- data.frame()
for (d in dirs) {
  tmp <- joinCSVFiles(d)
  gpt_qa <- rbind(gpt_qa, tmp)
}

gpt_qa %>% group_by(context_size) %>% summarise(m = mean(gpt_correct))
setwd("~/Documents/cut_attention/analysis")
write.csv(gpt_qa, 'gpt_acc_madlib1.csv')

dirs <- c('~/Dropbox/Cut Attention/Experiments/madlibs1_gpt3_10k/',
          '~/Dropbox/Cut Attention/Experiments/madlibs1_gpt3_1k/')
gpt_docs <- data.frame()
for (d in dirs) {
  tmp <- joinCSVFiles(d)
  gpt_docs <- rbind(gpt_docs, tmp)
}

gpt_docs %>% 
  group_by(context_size) %>% 
  summarise(m = mean(n_relevant_documents), 
            sem = sd(n_relevant_documents) / sqrt(n()))
setwd("~/Documents/cut_attention/analysis")
write.csv(gpt_qa, 'gpt_doc_madlib1.csv')

