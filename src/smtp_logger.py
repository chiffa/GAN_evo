from smtplib import SMTP
from logging.handlers import SMTPHandler
import logging
from datetime import datetime
from email.message import EmailMessage


#Send an email reporting the state of the code execution


local_host = 'lpdpc35.epfl.ch'
local_mail_account = 'andrei@lpdpc35.epfl.ch'
reporting_target_mail = 'andrei.kucharavy@epfl.ch'

mime_message = EmailMessage()
mime_message['From'] = local_mail_account
mime_message['To'] = reporting_target_mail


# smtp_server = SMTP(host=local_host)

logger = logging.getLogger()

mail_handler = SMTPHandler(mailhost=local_host,
                           fromaddr=local_mail_account,
                           toaddrs=reporting_target_mail,
                           subject="evoGAN runtime error")

mail_handler.setLevel(logging.ERROR)
logger.addHandler(mail_handler)


def successfully_completed(start_date, start_device):
    mime_message = EmailMessage()
    mime_message['From'] = local_mail_account
    mime_message['To'] = reporting_target_mail

    with SMTP(host=local_host) as smtp_server:
        mime_message['Subject'] = 'Run started on %s and %s has completed' % (start_date.isoformat(), start_device)
        mime_message.set_content('Run has completed successfully after %s minutes' % ((datetime.now() -
                   start_date).total_seconds()/60.))

        smtp_server.send_message(mime_message)


def smtp_error_bail_out():

    mime_message = EmailMessage()
    mime_message['From'] = local_mail_account
    mime_message['To'] = reporting_target_mail

    with SMTP(host=local_host) as smtp_server:
        mime_message['Subject'] = 'SMTPHandler error bail-out'
        mime_message.set_content("There was an error in the code and the logger's SMPTHandler bailed out.")
        print(smtp_server.noop())
        smtp_server.send_message(mime_message)


if __name__ == "__main__":

    # with SMTP(host=local_host) as smtp_server:
    #     mime_message['Subject'] = 'Test message from lpdpc28'
    #     mime_message.set_content('This is a test message')
    #     print(smtp_server.noop())
    #     smtp_server.send_message(mime_message)

    # logger.error('This is a test error log')

    smtp_error_bail_out()
