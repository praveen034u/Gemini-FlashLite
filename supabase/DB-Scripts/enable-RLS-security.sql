alter table message_history enable row level security;

create policy "Allow logged-in users to read/write their messages"
  on message_history
  for all
  using (auth.uid() = user_id::uuid);
