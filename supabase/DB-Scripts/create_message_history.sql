-- Enable RLS (optional but recommended)
alter table if exists message_history disable row level security;

-- Create table
create table if not exists message_history (
  session_id text not null,
  user_id text not null,
  role text not null,
  content text not null,
  timestamp timestamp with time zone default now()
);

-- Optional: Composite index for user/session queries
create index if not exists idx_user_session on message_history(user_id, session_id, timestamp);
