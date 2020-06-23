from smtplib import SMTP
from logging.handlers import SMTPHandler
import logging
from datetime import datetime

local_host = 'lpdpc28.epfl.ch'
local_mail_account = 'andrei@lpdpc28.epfl.ch'
reporting_target_mail = 'andrei.kucharavy@epfl.ch'

message = ("From: %s\r\nTo: %s\r\n\r\n" % (local_mail_account, ", ".join(reporting_target_mail)))

smtp_server = SMTP(host=local_host)

logger = logging.getLogger()
mail_handler = SMTPHandler(mailhost=local_host,
                           fromaddr=local_mail_account,
                           toaddrs=reporting_target_mail,
                           subject="evoGAN runtime error")

mail_handler.setLevel(logging.ERROR)
logger.addHandler(mail_handler)


def successfully_completed(start_date, start_device):
    global message
    message += 'Subject: Run started on %s and %s has completed' % (start_date.isoformat(), start_device)
    message += 'Run has completed successfully after %s total' % (datetime.now() -
               start_date).total_seconds()/60.
    smtp_server.sendmail(from_addr=local_mail_account,
                         to_addrs=reporting_target_mail,
                         msg=message)

if __name__ == "__main__":

    message += 'Subject: test message from lpdpc28'
    message += 'This is a test message for python email notification capabilities'

    print(smtp_server.noop())

    smtp_server.sendmail(from_addr=local_mail_account,
                         to_addrs=reporting_target_mail,
                         msg=message)